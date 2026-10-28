You are the bookkeeper of a small Belgian {industry} company. Your accounting software has its own chart of accounts, and you are exporting it.

Below are the {n_groups} groups of the Belgian standard chart (PCMN) that your company uses. Write the accounts your company has for these groups, exactly as they appear in your own system.

Rules:
- Write exactly {n_accounts} accounts in total: one for every group, plus a second account for some groups, the way companies split an account by site, project, bank, VAT rate or counterparty.
- Write the account names in {language_name}.
- Name the accounts the way a real {industry} company would in its own system: its own wording, abbreviations, product or site names and habits. Do not copy the standard names below; they are only there to tell you what each group means.
- Give every account its own account code. Codes must be unique. They do not have to follow the PCMN numbering; many companies use their own scheme.
- Set target_code to the code of the standard group the account belongs to, taken from the table.

| Group code | Standard name (English) | Standard name (company language) |
|---|---|---|
{groups_table}

Return only the JSON: one entry per account, with target_code, code and name.
