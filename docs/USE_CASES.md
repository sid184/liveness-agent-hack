# Finance deepfake use case: where it sits, how big, what matters

Research notes for the pitch, Sat 17 Oct 2026. Every number is linked. Where sources disagree or
are vendor-produced, it says so. Read the "Pitch lines" at the end first if short on time.

## 1. Where it sits, what it replaces, who benefits

### The moment

A payment instruction arrives on a call: a new payee, a transfer over threshold, a change of bank
details, an executive "authorising" something urgent. Today the person receiving it has three
options: trust the face and voice, run the company's callback procedure, or refuse. The Arup
case shows why the first option is dead: a finance worker in Hong Kong suspected the initial
email was phishing, joined a video call to check, and every other participant on the call,
including the CFO, was a deepfake built from public footage. HK$200m (about $25.6m) went out in
15 transfers before anyone checked with head office a week later
([CNN](https://abc17news.com/money/cnn-business-consumer/2024/05/16/british-engineering-giant-arup-revealed-as-25-million-deepfake-scam-victim/),
[Business Insider](https://businessinsider.nl/an-engineering-giant-behind-apples-hq-and-the-sydney-opera-house-lost-25-million-when-scammers-tricked-an-employee-with-a-deepfake-of-a-senior-exec/)).
The video call *was* the verification step, and it was the thing that defeated him.

### What we replace or augment

The standard controls, per treasury guidance
([Adaptive Security](https://www.adaptivesecurity.com/blog/deepfake-fraud-prevention-finance-teams),
[Connext](https://connextglobal.com/accounts-payable-fraud-prevention-deepfake-authorization-protocol/),
[Phacet](https://www.phacetlabs.com/blog/pre-payment-fraud-prevention-controls)):

| Control today | Weakness | What our check does instead |
|---|---|---|
| Callback to a number on file | Slow (minutes to hours to reach an exec), and routinely skipped when the request feels urgent or senior | Ten seconds, inside the call, with no social cost: the system asks, not the clerk |
| Dual authorisation | The second approver gets the same deepfake call (reported in 2026 cases) | Each approver's call can carry its own live check |
| Shared passphrase or "a question only they'd know" | Works once (Ferrari, July 2024: an exec asked the fake CEO which book he'd recommended that week, call ended), then has to be reset | Challenges are physical and randomised; there is nothing to leak |
| "Hang up and call back" | Breaks the meeting; nobody does it to their CFO | The check is a normal part of the meeting |

Ferrari source: [Fortune/Bloomberg](https://fortune.com/2024/07/27/ferrari-deepfake-attempt-scammer-security-question-ceo-benedetto-vigna-cybersecurity-ai).

### Who benefits, in order

1. **The person receiving the instruction**: accounts payable clerk, treasury analyst, EA. They
   are the ones who get blamed and, in the Arup case, prosecuted (the employee was reportedly
   convicted). Medius found 87% of finance professionals would make a payment if "called" by
   their CEO or CFO
   ([CFO Dive](https://www.cfodive.com/news/deepfake-scams-escalate-hitting-53-percent-of-businesses/725836)).
   The product gives them a way to push back that costs them nothing socially.
2. **Controllers, treasurers and CFOs** who own payment policy and the loss when it fails. The
   check fires only on high-risk actions they define, so it's a policy they can actually enforce.
3. **Bank fraud and wire-verification teams**, who call clients to confirm large wires and face
   the same impersonation problem from the other side. In the UK, banks have reimbursed APP fraud
   victims under mandatory rules since October 2024, so the loss sits with them
   ([UK Finance via FinTech Futures](https://www.fintechfutures.com/financial-crime-fraud-prevention/uk-payment-fraud-losses-top-1-28bn-in-2025-new-uk-finance-report-finds)).
4. **The executive being impersonated**, who gets a frictionless way to prove it's really them.

Adjacent markets with the same mechanism: private banking client instructions over video, IT and
HR helpdesks (credential resets), insurance claims over video.

## 2. Is it big enough, and why this tech

### The envelope

Deepfake-on-video losses are not reported as a separate line anywhere yet. The honest envelope is
business email compromise and CEO fraud, where the deepfake call is the new delivery channel.

| Figure | Value | Source and caveat |
|---|---|---|
| BEC losses, US, 2024 | $2.77bn from 21,442 complaints, about $130k per incident | FBI IC3 via [Proofpoint](https://www.proofpoint.com/us/blog/email-and-cloud-threats/email-attacks-drive-record-cybercrime-losses-2024), [NACHA](https://www.nacha.org/news/fbis-ic3-finds-almost-85-billion-lost-business-email-compromise-last-three-years) |
| BEC losses, US, 2025 | $3.05bn from 24,768 complaints | IC3 as cited by [DeepStrike](https://deepstrike.io/blog/business-email-compromise-statistics-2026); secondary, verify against the FBI report |
| Organisations hit by payments fraud | 79% in 2024, 76% in 2025 | [AFP 2025](https://www.financialprofessionals.org/about/learn-more/press-releases/Details/survey-79-percent-of-organizations-were-victims-of-attempted-or-actual-payments-fraud-activity-in-2024), [AFP 2026](https://www.financialprofessionals.org/about/learn-more/press-releases/Details/over-75-percent-of-us-firms-experienced-payments-fraud-in-2025-while-ai-adoption-for-fraud-mitigation-lags) |
| Organisations with an actual BEC loss | 36% in 2024 | [AFP 2025 highlights (Truist)](https://truist.com/content/dam/truist-bank/us/en/documents/info/cci/2025-afp-payments-fraud-control-survey-report-key-highlights.pdf) |
| Finance professionals targeted by deepfakes | 53% targeted, 43% fell victim (1,533 US and UK respondents, 2024) | [Medius via CFO Dive](https://www.cfodive.com/news/deepfake-scams-escalate-hitting-53-percent-of-businesses/725836); vendor survey |
| Gen-AI-enabled fraud losses, US | $12.3bn in 2023 to a projected $40bn by 2027 | [Deloitte](https://www.deloitte.com/us/en/insights/industry/financial-services/deepfake-banking-fraud-risk-on-the-rise.html); 2024 forecast |
| UK payment fraud 2025 | £1.28bn total; APP fraud £576m (+19%), of which £75.6m from businesses | [UK Finance via FinTech Futures](https://www.fintechfutures.com/financial-crime-fraud-prevention/uk-payment-fraud-losses-top-1-28bn-in-2025-new-uk-finance-report-finds) |
| Deepfake attempts in contact centres | up more than 1,300% in 2024, from about one a month to seven a day | [Pindrop](https://www.pindrop.com/wp-content/uploads/2025/07/Voice-Intelligence-and-Security-Report-Report-June-2025.pdf); vendor; Pindrop's own article cites 680% |
| Face-swap attacks on identity verification | up 300% vs 2023; native virtual-camera injection up 2,665%; about 24,000 sellers of attack tools | [iProov 2025](https://www.iproov.com/reports/threat-intelligence-report-2025-remote-identity-attack); vendor |
| Enterprises that will stop trusting identity verification alone | 30% by 2026 | [Gartner, 2024 forecast](https://futurecio.tech/deepfakes-to-render-identity-verification-and-authentication-solutions-inefficient/) |

### What a buyer saves

Two things, and the second matters more than the first.

**Loss avoided.** A typical organisation faces roughly a one-in-three chance of an actual BEC
loss in a year, and the average reported loss is about $130k. The tail is Arup: $25m in one
afternoon. For a bank under UK reimbursement rules, every successful impersonation of a customer
is a direct hit.

**Friction removed.** This is the part that gets the product bought rather than admired. A
callback procedure done properly costs two people twenty minutes and often a day's delay waiting
for an executive, which is exactly why it gets skipped. (Assumption, not a sourced figure.) A ten
second in-call check that only fires on the 5% of payments that are high-risk removes the reason
to skip. The product wins when it replaces a worse step, not when it adds one.

### Why this tech and not a detector

- **Passive detection is an arms race it is losing.** Deloitte describes deepfake generators that
  self-check against detectors. iProov says a technique first seen in December 2024 bypasses most
  current remote identity verification. Gartner's 30% figure is about exactly this.
- **Challenge-response moves the burden.** The attacker has to render a novel physical interaction
  live, in real time. NYU's GOTCHA paper (IEEE Euro S&P 2024) tested eight challenge types against
  real-time deepfakes on 47 subjects and found occlusion and facial deformation the most effective,
  with the "vanishing object" artifact the one humans spotted most. Human judges reached 88.6% AUC,
  an automated scorer 80.1% ([arXiv 2210.06186](https://arxiv.org/abs/2210.06186v4)). Our object
  pass challenge is that finding, productised.
- **What we add over GOTCHA.** They trained a fidelity classifier. We use a physical-world
  reasoning model (Cosmos) that explains *why* in words, a detector (YOLO) that measures what
  happened and cannot be talked round, and a memory of past sessions. The explanation is the
  product: a finance person acts on "the phone never crossed the face", not on "0.3".
- **Complementary, not competing, with the one in-call vendor.** Beyond Identity's RealityCheck
  puts a badge on Zoom participants proving the *device* and *account* are authenticated
  ([Zoom marketplace](https://marketplace.zoom.us/apps/ZgT5jp7xTfyqGks_h8M-Jg)). It proves the
  laptop; it does not prove the person in front of it is live right now, and it does nothing on
  an external call. Onboarding liveness vendors (iProov, Onfido, Jumio) verify once at sign-up,
  not at the moment of action.

## 3. Features that matter most, and why this is a data problem

Ranked by how much they change the outcome for the people in section 1.

1. **Fire only on high-risk actions.** The policy gate reads transaction data: amount, payee
   novelty, bank-detail change, who is asking. Zero friction on 95% of payments is what makes the
   5% acceptable. This is a data feature before it is a video feature.
2. **A verdict with evidence, in plain English.** Clip plus reasoning plus the measurements.
   Three audiences need it: the clerk deciding now, the fraud team afterwards, and the lawyer.
   Deepfake-induced payments sit in a legal grey zone between authorised and unauthorised, so
   insurers and banks dispute them
   ([Wright Law](https://www.wrightlawaz.com/newsroom/articles/2026/ai-deepfake-wire-fraud-ucc-4a-2026/)).
   An evidence record of the check is what settles it.
3. **Randomised, rotating challenges.** The Ferrari question worked once. A library of physical
   challenges with varied parameters never repeats and has nothing to leak.
4. **Depersonalise the ask.** "The system requires a live check for payments over £50k" is
   something a junior can say to a CFO. "I don't believe you" is not. The 87% number is a social
   problem, and this is the fix.
5. **Memory and search across sessions.** The same attacker tries the same trick on several
   firms. Semantic search over past sessions and known attacks turns each catch into protection
   for the next target. Across customers this is a network effect; inside one firm it is how the
   agent gets sharper.
6. **Cheap failure.** An inconclusive result issues a different challenge; two misses fall back to
   the callback. A false reject costs ten seconds, a false accept costs the money.
7. **Signed capture path.** Device attestation so a virtual camera can't inject footage. iProov's
   2,665% injection figure says this is where attackers are heading. Out of scope today; it is the
   obvious partnership with the RealityCheck-style vendors.

### Why it's a data product

- **It consumes transaction data** to decide when to fire and how hard a challenge to issue.
- **It produces the scarcest dataset in this field.** Every session is a labelled clip: a real
  person doing a known challenge, or a confirmed attack. GOTCHA had to recruit 47 people to build
  theirs. A deployed product builds one every day, and that dataset is what recalibrates the
  thresholds and trains the next detector.
- **It makes stored video useful.** The hackathon brief is video sitting unwatched. Verification
  sessions become a searchable evidence archive: when a fraud surfaces a week later, as at Arup,
  "every call in which this payee was discussed" is one query, with the clips attached.

## Pitch lines

- "In 2024 a finance worker joined a video call to check a suspicious email. Everyone on the call
  was fake. $25 million left that afternoon."
- "Half of finance teams have been targeted with deepfakes. 87% say they'd pay if the CEO called.
  The face and the voice are no longer evidence."
- "We don't try to spot the fake. We ask the caller to do something a real-time fake can't render,
  and an agent that reasons about the physical world watches them do it."
- "Ten seconds, inside the call, only on the payments that matter. It replaces the callback nobody
  makes."
- "Every check is logged with its evidence, so the next verdict is sharper and the lawyers have
  something to read."
