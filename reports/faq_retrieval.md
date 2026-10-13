# FAQ retrieval (BM25)

Produced by `cva-faq-eval`. Questions: `knowledge/faq_eval.yaml`.

Answerable questions: 36 · unanswerable: 4

| Hit@1 | Hit@3 | MRR@3 |
|---|---|---|
| 0.833 | 0.944 | 0.875 |

## Score threshold for "no answer"

| min score | answerable: correct top-1 kept | unanswerable: rejected |
|---|---|---|
| 0.0 | 0.833 | 0.500 |
| 1.0 | 0.833 | 0.500 |
| 2.0 | 0.833 | 0.500 |
| 3.0 | 0.667 | 0.750 |
| 4.0 | 0.583 | 0.750 |
| 5.0 | 0.444 | 0.750 |

## Misses (gold not ranked first)

- "what time do you open tomorrow morning" → gold `opening-hours`, got `cancelling-or-rescheduling` (rank 3)
- "until when can I come in on a friday" → gold `opening-hours`, got `fever-or-infection-symptoms` (rank 3)
- "who are the doctors at the clinic" → gold `doctors-and-staff`, got `contacting-the-clinic` (rank None)
- "I have several things to talk about with the doctor, what should I book" → gold `appointment-types-and-length`, got `address-and-directions` (rank None)
- "do you take private patients" → gold `insurance-and-payment`, got `registering-as-a-new-patient` (rank 3)
- "can I leave a message for Dr Weber" → gold `leaving-a-message-for-a-doctor`, got `doctors-and-staff` (rank 2)

## Unanswerable questions: top score

- "do you sell concert tickets" → `-` (0.00)
- "what's the weather going to be like tomorrow" → `what-to-bring-to-an-appointment` (2.97)
- "can you recommend a good dentist" → `-` (0.00)
- "is the pharmacy next door open on sundays" → `opening-hours` (5.75)
