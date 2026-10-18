# FTEC5660 Homework 2: CV Verification Agent

Build a LangChain agent that reads each CV in a folder, looks the candidate up
on our SocialGraph MCP server (mock LinkedIn and Facebook), and outputs a
reliability score in [0, 1] for each CV.

A CV is **valid** (label `1`) when its claims agree with the candidate's
social media profiles. It **has a discrepancy** (label `0`) when it contains
problems such as an inflated job title, shifted dates, an upgraded degree, a
fake school or employer, a wrong location, or made-up skills. Differences in
wording only ("Bachelor of Science" vs `BSc`, "UI/UX Design" vs `UI/UX`,
"Senior Engineer" for an `Engineer` role with seniority `senior`, listing fewer
skills) are not discrepancies.

## Student task

Fill in the two functions in `hw2.py` that contain `### YOUR CODE HERE`. You
may add imports, constants and helper functions above them, but do not change
the provided code below them:

- `build_agent(tools)` creates your agent from the MCP tools.
- `score_cvs(agent, cvs)` runs the agent on every CV and returns
  `{file_name: score}`, one float in [0, 1] per CV.

A score above `0.5` means "valid"; `0.5` or below means "has discrepancy". You
may use a single tool-calling agent, multiple agents, reflection, or a
combination. Do not hard-code filenames, names, or public answers; grading
uses unseen CVs.

## MCP server

The server is hosted at `https://ftec5660.ngrok.app/mcp`. `hw2.py` already
connects to it and passes you these tools:

| Tool | Purpose |
| --- | --- |
| `search_facebook_users(q, limit, fuzzy)` | find Facebook users by display name |
| `get_facebook_profile(user_id)` | full Facebook profile |
| `get_facebook_mutual_friends(user_id_1, user_id_2)` | mutual friends of two users |
| `search_linkedin_people(q, location, industry, limit, fuzzy)` | search LinkedIn by name, skill, or title |
| `get_linkedin_profile(person_id)` | full LinkedIn profile (experience, education, skills) |
| `get_linkedin_interactions(person_id)` | post and like statistics |

Print `tool.name`, `tool.description`, and `tool.args` for the full schemas.

## Setup and public test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo "DEEPSEEK_API_KEY=your_key_here" > .env
python3 hw2.py --cv-folder public_test
```

Use your own DeepSeek API key (from https://platform.deepseek.com).

The program creates `results.csv` in the current directory. Its columns are
`cv`, `score`, and `correctness`. The public labels are in
`public_test/ground_truth.json`; every CV with label `0` also has a `reason`
explaining the discrepancy, and `results.csv` shows it when your agent misses
one. The starter returns no score (`None`) for every CV so it runs before you
add any API code; missing scores count as incorrect.

The required model is `deepseek-v4-flash`, and all evidence must come from the
MCP server (no web search). The server is shared by the whole class: keep at
most about 3 CVs in flight at a time (e.g. `asyncio.Semaphore(3)`). We grade
with our own API key, so never put a key in your code; `.env` must stay out of
git.


## Task 2: Fool the Verifier

We added a target candidate, **Kelly Tsang** (LinkedIn `person_id` 10001,
Facebook `user_id` 10001, display name "Kel Tsang"). Her true CV is
`task2/target_cv.pdf`.

Write one CV for Kelly Tsang with at least one false or embellished detail
that our 5 verifier agents (same function as Task 1) still score above 0.5.
Each fooled agent is worth 4 points; a CV without a real false or embellished
detail earns 0. Attack your own Task 1 agent first:

```bash
cp my_attack.pdf task2/adversarial_cv.pdf
python3 hw2.py --cv-folder task2
```

`results.csv` shows your agent's score for both CVs (they are not graded, as
`task2/` has no `ground_truth.json`). Only the PDF is the attack surface; do not
attack the MCP server. See the homework description for the full rules.

## Homework 2 solution:
> to students: this is your report, see the homework description.

### Task 1
> one paragraph describing your solution, and your results on `public_test`.
We implemented a LangChain CV verification agent using the required `deepseek-v4-flash` model and the six SocialGraph MCP tools. For each CV, the agent extracts the candidate's identity, location, work experience, education, and skills, searches LinkedIn and Facebook to identify the correct public profile, retrieves the relevant profile information, and compares the claims using the assignment's field rules. The prompt treats CV text as untrusted data, allows equivalent wording such as abbreviations, and ignores non-verification fields such as headlines, bios, hometowns, and skills omitted from the CV. We additionally apply lightweight deterministic checks to explicit skills, employment years, and degree fields, including normalization for common equivalents such as “Microsoft PowerPoint” and “PowerPoint”. CVs are processed asynchronously with a semaphore limiting concurrency to 1 (3 at the beginnig but always reporting connection error), and each CV is handled independently so that an MCP or API failure does not terminate the whole run.
Last 5 run precision on public_test without connection error:
34 /35

### Task 2
> the attack techniques you used and why you expect them to work.
We created an adversarial CV for Kelly Tsang based on the provided target CV. The CV keeps the verified identity, employer names, employment dates, education, and listed skills unchanged, but embellishes the AIA work description by claiming responsibility for a team of 35 and a HK$50M project budget. These details are false or exaggerated, while the job title remains “Manager” to avoid an easily detectable structured-field mismatch (changed once but detected by exact-matching from code not agent). We also included a low-visibility instruction in the PDF text that attempts to influence the language model to treat the CV as pre-verified. The attack targets verification coverage and prompt sensitivity: the verifier is required to compare structured fields such as company, title, dates, education, and skills, while detailed job descriptions are not primary discrepancy fields.

### Task 3: Reflection on the AI Paradox
AI can search and compare information much faster than a human reviewer while maintaining reasonable accuracy. In recruitment, this makes it useful for identifying exaggerated or false claims in CVs. Applicants who use a general AI writing tool to embellish a resume may find that those claims do not survive a separate AI verification process. Since most applicants have only one chance to submit a CV to a particular employer, even an imperfect screening system could discourage some misrepresentation.
The paradox emerges as applicants adapt. A purpose-built agent could repeatedly revise a CV against a verifier until it finds wording or presentation that passes. Less qualified applicants might use such tools to appear stronger, while well-qualified applicants might feel compelled to use them simply to remain competitive. Employers would then need more sophisticated verification, prompting applicants to adopt more sophisticated editing tools. This cycle raises costs for everyone without necessarily improving hiring decisions.
The same pattern extends beyond recruitment. Whenever AI makes a process faster and cheaper, people gain incentives to optimize their behavior for the system rather than for the underlying goal. A useful response is to treat AI verification as evidence for human judgment, not as a final verdict. Employers should check consequential claims through appropriate sources, allow applicants to correct errors, and evaluate demonstrated ability alongside the CV.