You are an experienced Belgian accountant. Below are {n_accounts} account(s) from one company's chart of accounts, in the company's own order. Map each one to its group in the Belgian standard chart of accounts (PCMN / MAR, 3-digit level).

Every account comes with its own short list of candidate groups. For each account, give:
- choices: exactly {n_choices} different groups from that account's own candidate list, most likely first, each with the probability (0 to 100) that it is the right group. A person reviewing the account chooses from this list, so always give {n_choices}, even when the lower ones are unlikely. If it is likely that none of the candidates fits, add "none" as an extra choice with its probability. The probabilities of one account should add up to 100 at most.
- reason: one short sentence explaining the most likely choice, naming a close alternative if there is one.

Things to know:
- Account names are in Dutch, French, English or a mix, and often abbreviated.
- The company code may follow the company's own numbering. A code that starts like a standard group is a strong hint, but accounts are sometimes filed under the wrong number, so check it against the name.
- The accounts just above and below show where in the company's chart the account sits.

{accounts}

Return only the JSON: one answer per account, with its account label, choices and reason.
