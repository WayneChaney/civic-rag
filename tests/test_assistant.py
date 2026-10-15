import unittest
from pathlib import Path

from assistant import BM25, ask, load_corpus, scrub_pii


CORPUS = Path(__file__).resolve().parents[1] / "_private_corpus" / "pages.json"


class AssistantTests(unittest.TestCase):
    def test_scrubs_common_pii(self):
        value = scrub_pii("My email is person@example.com, card 4111 1111 1111 1111, and phone 555-123-4567.")
        self.assertNotIn("person@example.com", value)
        self.assertNotIn("4111", value)
        self.assertNotIn("555-123", value)

    @unittest.skipUnless(CORPUS.exists(), "private evaluation corpus is not present")
    def test_bm25_finds_leaf_and_rubbish_pages(self):
        data = load_corpus(CORPUS)
        page_ids = {page["url"]: index for index, page in enumerate(data["pages"])}
        retriever = BM25(data["chunks"])
        leaf = [page_ids[chunk["url"]] for chunk, _ in retriever.search("When is leaf collection?", top=3)]
        rubbish = [page_ids[chunk["url"]] for chunk, _ in retriever.search("When is rubbish pickup?", top=3)]
        self.assertIn(35, leaf)
        self.assertIn(37, rubbish)

    def test_medical_question_refuses_without_a_provider(self):
        result = ask("Can you diagnose my medical symptoms?", BM25([]))
        self.assertTrue(result["refused"])
        self.assertEqual(result["sources"], [])


if __name__ == "__main__":
    unittest.main()
