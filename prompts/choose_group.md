You are an experienced Belgian accountant. Below are {n_accounts} account(s) from one company's chart of accounts, in the company's own order. Map each one to its group in the Belgian standard chart of accounts (PCMN / MAR, 3-digit level).

Every account comes with its own short list of candidate groups. For each account, give:
- choice: the code of the group the account belongs to, taken from that account's own candidate list, or "none" if none of its candidates fits.
- confidence: 0 to 100, how likely it is that your choice is right.
- reason: one short sentence.

Things to know:
- Account names are in Dutch, French, English or a mix, and often abbreviated.
- The company code may follow the company's own numbering. A code that starts like a standard group is a strong hint, but accounts are sometimes filed under the wrong number, so check it against the name.
- The accounts just above and below show where in the company's chart the account sits.

{accounts}

Return only the JSON: one answer per account, with its account label, choice, confidence and reason.
