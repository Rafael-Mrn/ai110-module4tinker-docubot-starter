"""
Core DocuBot class responsible for:
- Loading documents from the docs/ folder
- Building a simple retrieval index (Phase 1)
- Retrieving relevant snippets (Phase 1)
- Supporting retrieval only answers
- Supporting RAG answers when paired with Gemini (Phase 2)
"""

import os
import glob
import string
import re

# Common English filler words that carry no retrieval signal. Removing them
# stops queries like "Which endpoint returns all users?" from awarding points
# for "which"/"all", which would otherwise pad irrelevant documents.
STOPWORDS = {
    "a", "an", "and", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "to", "in", "on", "at", "for", "from", "by", "with", "as", "that",
    "this", "these", "those", "it", "its", "i", "you", "he", "she", "they",
    "we", "do", "does", "did", "how", "what", "when", "where", "which", "who",
    "why", "all", "any", "some", "no", "not", "or", "if", "then", "else",
    "there", "here", "so", "such", "can", "will", "would", "should", "could",
}

# Guardrail settings. Refusing to answer is a feature: a clear "I don't know"
# is safer than a confident guess from an irrelevant chunk.
MIN_COVERAGE = 2  # For multi-word queries, the best chunk must match this many
                  # distinct query words. The bar is applied relative to query
                  # length (see has_sufficient_evidence), so single-word queries
                  # still only need their one word.
REFUSAL_MESSAGE = "I do not know based on these docs."


class DocuBot:
    def __init__(self, docs_folder="docs", llm_client=None):
        """
        docs_folder: directory containing project documentation files
        llm_client: optional Gemini client for LLM based answers
        """
        self.docs_folder = docs_folder
        self.llm_client = llm_client

        # Load documents into memory
        self.documents = self.load_documents()  # List of (filename, text)

        # Split each document into smaller sections. Retrieval works on these
        # chunks so answers cite a relevant paragraph, not a whole file.
        self.chunks = self.chunk_documents(self.documents)  # List of (filename, text)

        # Build a retrieval index over the chunks (implemented in Phase 1)
        self.index = self.build_index(self.chunks)

    # -----------------------------------------------------------
    # Document Loading
    # -----------------------------------------------------------

    def load_documents(self):
        """
        Loads all .md and .txt files inside docs_folder.
        Returns a list of tuples: (filename, text)
        """
        docs = []
        pattern = os.path.join(self.docs_folder, "*.*")
        for path in glob.glob(pattern):
            if path.endswith(".md") or path.endswith(".txt"):
                with open(path, "r", encoding="utf8") as f:
                    text = f.read()
                filename = os.path.basename(path)
                docs.append((filename, text))
        return docs

    # -----------------------------------------------------------
    # Chunking (split documents into retrievable sections)
    # -----------------------------------------------------------

    def chunk_documents(self, documents):
        """
        Split each document into heading-aware sections and flatten them into
        a single list of (filename, section_text) tuples.

        Strategy: start a new chunk at every markdown heading line (one that
        begins with '#'), keeping the heading together with the body text that
        follows it. This is better than blank-line splitting for these docs,
        which would sever a heading like "## User Data Endpoints" from the
        paragraph describing that endpoint. Any text before the first heading
        (a preamble) becomes its own chunk. The filename rides on every chunk
        so answers can still cite it.
        """
        chunks = []
        for filename, text in documents:
            # Split right before each heading line; the zero-width lookahead
            # keeps the heading at the start of its section.
            for section in re.split(r"(?m)^(?=#{1,6}\s)", text):
                section = section.strip()
                if section:
                    chunks.append((filename, section))
        return chunks

    # -----------------------------------------------------------
    # Tokenization (shared by indexing and scoring)
    # -----------------------------------------------------------

    def tokenize(self, text):
        """
        Split text into a list of normalized tokens.

        Used by BOTH build_index and score_document so that a query word
        and an indexed word are normalized the exact same way (otherwise
        "token," in a query would never match the indexed "token").

        Normalization pipeline:
          1. lowercase and split on whitespace
          2. strip surrounding punctuation, drop empties
          3. drop stopwords (filler words with no retrieval signal)
          4. stem simple plurals so "endpoints" matches "endpoint"
        """
        tokens = []
        for raw in text.lower().split():
            word = raw.strip(string.punctuation)
            if not word or word in STOPWORDS:
                continue
            tokens.append(self._stem(word))
        return tokens

    def _stem(self, word):
        """
        Light, dependency-free plural stemmer. Not linguistically complete —
        just enough to make singular/plural forms collide in the index so a
        query for "endpoint" matches a doc that says "endpoints".
        """
        if word.endswith("ies") and len(word) > 4:
            return word[:-3] + "y"          # queries -> query
        if word.endswith("ses") and len(word) > 4:
            return word[:-2]                # classes -> class
        if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
            return word[:-1]                # endpoints -> endpoint
        return word

    # -----------------------------------------------------------
    # Index Construction (Phase 1)
    # -----------------------------------------------------------

    def build_index(self, chunks):
        """
        Build a tiny inverted index mapping each lowercase word to the chunks
        it appears in.

        Deviation from the original docstring: values are *chunk indices*
        (positions in self.chunks), not filenames. A chunk is the unit of
        retrieval now, and several chunks share a filename, so a filename is
        no longer a unique key. Indices point straight back to (filename, text).

        Example structure:
        {
            "token": [0, 4],      # chunk 0 and chunk 4 contain "token"
            "database": [7]
        }
        """
        index = {}
        for i, chunk in enumerate(chunks):
            text = chunk[1]
            # Use a set so each word maps to a given chunk only once.
            for word in set(self.tokenize(text)):
                index.setdefault(word, []).append(i)
        return index

    # -----------------------------------------------------------
    # Scoring and Retrieval (Phase 1)
    # -----------------------------------------------------------

    def score_document(self, query, text):
        """
        Return a relevance score for how well the text matches the query.

        Baseline used here: distinct-presence coverage -- the number of
        distinct query words that appear in the text. See match_frequency
        for the tiebreaker used alongside this in retrieve.
        """
        # Distinct-presence scoring: how many *distinct* query words appear
        # in the text. Uses sets so repeats on either side count once, and
        # the same tokenizer as the index so matches line up.
        query_words = set(self.tokenize(query))
        text_words = set(self.tokenize(text))
        return len(query_words & text_words)

    def match_frequency(self, query, text):
        """
        Total number of times the query's words occur in the text.

        Used ONLY as a tiebreaker in retrieve, never as the primary score:
        distinct-presence coverage decides ranking first, so a single very
        frequent word can't outrank a document that covers more of the query
        (avoiding the classic raw-frequency failure). It only separates
        documents that already tied on coverage.
        """
        query_words = set(self.tokenize(query))
        return sum(1 for word in self.tokenize(text) if word in query_words)

    def retrieve(self, query, top_k=3):
        """
        Use the index and scoring function to select the top_k relevant
        chunks (paragraph-sized sections), not whole documents.

        Return a list of (filename, text) sorted by relevance descending.
        """
        # 1) Use the index to gather only the chunks containing >=1 query word,
        #    so we score candidates instead of every chunk in the corpus.
        candidates = set()
        for word in set(self.tokenize(query)):
            candidates.update(self.index.get(word, []))

        if not candidates:
            return []

        # 2) Rank by (coverage, frequency): distinct-word coverage is the
        #    primary key, total match frequency only breaks ties. Iterating
        #    candidates in index order keeps ties in original load order.
        scored = []
        for i in sorted(candidates):
            filename, text = self.chunks[i]
            coverage = self.score_document(query, text)
            frequency = self.match_frequency(query, text)
            scored.append(((coverage, frequency), filename, text))
        scored.sort(key=lambda item: item[0], reverse=True)

        return [(filename, text) for _, filename, text in scored[:top_k]]

    # -----------------------------------------------------------
    # Guardrail (decide whether we have enough evidence to answer)
    # -----------------------------------------------------------

    def has_sufficient_evidence(self, query, snippets):
        """
        Decide whether the retrieved snippets are strong enough to answer,
        rather than guessing from an incidental match.

        Rule: the best snippet must match at least min(MIN_COVERAGE, N)
        distinct query words, where N is the number of query content words.
        - Single-word query (N=1): needs its one word -> full coverage.
        - Multi-word query (N>=2): needs MIN_COVERAGE words, so a lone
          coincidental match is rejected.

        Returns True if we should answer, False if DocuBot should refuse.
        """
        if not snippets:
            return False  # zero chunks matched any query word

        query_tokens = set(self.tokenize(query))
        if not query_tokens:
            return False  # empty or stopword-only query -> no content to match

        required = min(MIN_COVERAGE, len(query_tokens))
        # retrieve() returns snippets best-first, so snippets[0] is the strongest.
        _, best_text = snippets[0]
        return self.score_document(query, best_text) >= required

    # -----------------------------------------------------------
    # Answering Modes
    # -----------------------------------------------------------

    def answer_retrieval_only(self, query, top_k=3):
        """
        Phase 1 retrieval only mode.
        Returns raw snippets and filenames with no LLM involved.
        """
        snippets = self.retrieve(query, top_k=top_k)

        if not self.has_sufficient_evidence(query, snippets):
            return REFUSAL_MESSAGE

        formatted = []
        for filename, text in snippets:
            formatted.append(f"[{filename}]\n{text}\n")

        return "\n---\n".join(formatted)

    def answer_rag(self, query, top_k=3):
        """
        Phase 2 RAG mode.
        Uses student retrieval to select snippets, then asks Gemini
        to generate an answer using only those snippets.
        """
        if self.llm_client is None:
            raise RuntimeError(
                "RAG mode requires an LLM client. Provide a GeminiClient instance."
            )

        snippets = self.retrieve(query, top_k=top_k)

        # Guard BEFORE calling the LLM so it never sees weak context and
        # can't hallucinate a confident answer from an irrelevant snippet.
        if not self.has_sufficient_evidence(query, snippets):
            return REFUSAL_MESSAGE

        return self.llm_client.answer_from_snippets(query, snippets)

    # -----------------------------------------------------------
    # Bonus Helper: concatenated docs for naive generation mode
    # -----------------------------------------------------------

    def full_corpus_text(self):
        """
        Returns all documents concatenated into a single string.
        This is used in Phase 0 for naive 'generation only' baselines.
        """
        return "\n\n".join(text for _, text in self.documents)
