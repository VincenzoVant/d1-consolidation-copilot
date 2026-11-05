You are an experienced Belgian accountant. Below are {n_groups} groups of the Belgian standard chart of accounts (PCMN / MAR), at the 3-digit level, with their official names and the group they belong to.

For every group, write:
- description: one plain English sentence saying what is booked in this group, in the words an accountant would use to explain it to a new colleague. Do not repeat the official name; say what it means in practice.
- words_nl, words_fr, words_en: 3 to 6 short words or phrases that Belgian bookkeepers commonly use in account names for this group, in Dutch, French and English. Use everyday ledger wording and common abbreviations, not the official name again. Keep each one short (one to four words) and generic: no company, brand, bank or person names.

Rules:
- Write exactly one entry for every group in the table, with its group code, and no other entries.
- Keep groups with similar names apart: say what makes this group different from its neighbours (for example a balance-sheet provision versus the cost that builds it up).

| Group code | Official name (English) | Official name (Dutch) | Official name (French) | Belongs to |
|---|---|---|---|---|
{groups_table}

Return only the JSON: one entry per group, with code, description, words_nl, words_fr and words_en.
