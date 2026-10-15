# Evaluation results

Run on a small Ohio village's 54-page public website with `ollama`. Page text and URLs are intentionally left out.

| What was measured | Result |
| --- | ---: |
| Right page in the top 3 search results | **86.7%** (26/30) |
| Right page anywhere in the top 6 | 93.3% |
| Right page ranked first | 53.3% |
| Answered and cited the correct page | **83.3%** (25/30) |
| Said "I can't find that" when the site did have it | 1/30 |
| Off-topic questions declined, blind set (separate agent) | **13/15** |
| Off-topic questions declined, builder set | 6/6 |
| Median / p90 answer time | 9.3s / 13.5s |
| Model cost | $0.00 (local model)|

A page only gets cited when the model actually answered. If it says the passages don't cover the question, citations are dropped, so a non-answer can't pass as grounded.

## Every miss

- `What services does the service department provide?`: answered, but cited the wrong page
- `Are there any current job openings?`: answered, but cited the wrong page (retrieval missed the page)
- `What can I put in recycling?`: said it couldn't answer, but the site does
- `What help is available for older residents?`: answered, but cited the wrong page (retrieval missed the page)
- `Can residents rent a truck?`: answered, but cited the wrong page
- `how do I get a copy of my birth certificate`: blind off-topic question, answered anyway
- `how do I register to vote or find out where my polling place is for the November election`: blind off-topic question, answered anyway
