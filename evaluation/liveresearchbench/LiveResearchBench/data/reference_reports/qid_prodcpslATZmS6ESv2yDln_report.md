# Project Management Tools Comparison (as of Sep 16, 2025) — Evidence-Based Coverage for Notion and ClickUp

## Scope and methodology
- This report covers Notion and ClickUp in depth because the research package provided primary-source documentation for these two vendors only. All links are official vendor help centers, product docs, release notes, pricing pages, integration directories, or developer docs.
- All sources were accessed on 2025-09-16.
- Where a vendor does not publish a figure (e.g., an exact integration count), that absence is noted with a primary-source citation where possible.

## Notion

### 1) Customization capabilities and template flexibility
- Custom fields (database properties): Title, Text, Number, Select, Status, Multi-select, Date, Formula, Relation, Rollup, Person, File(s), Checkbox, URL, Email, Phone, Created/Last edited time/by, Button, ID; Button is a property that can run actions on click [1]. Views include Table, Board (Kanban), Timeline, Calendar, List, Gallery [2]. Chart view (bar, line, donut) supports dashboards; Free plan can create one chart; paid plans can create unlimited [3].
- Filters/sorts/groups are first-class for database views [2].
- Automations (database-level): triggers (Page added, Property edited with conditions, Recurring), actions (Edit property, Add page to, Edit pages in other DBs, Send notification/mail/webhook, Slack); database automations are paid; Free can create Slack notifications only [4].
- Button property and “database buttons” allow multi-action workflows (e.g., approve) [1][5].
- Forms: built-in, available on all plans; connect to a database; share to workspace or web; optional anonymous responses; support automations [6].
- Embeds and link previews (e.g., Slack, Jira, GitHub) supported [7][8].
- Roles/permissions: Admins, Members, Guests, Groups; granular database permission “Can edit content” lets users edit rows without changing schema/views [9][10][11].
- Templates: users can submit/publish templates; official gallery lists “20,000+ Notion templates”; duplication supported; export options include HTML/Markdown/CSV (databases); bulk PDF export on Business plan [12][13][14][15][16].

### 2) Collaboration features
- Real-time co-editing with multi-user presence; multiple users can edit simultaneously [17].
- Comments, @mentions (people, groups, pages, dates), reminders; inbox and page notifications included [18][19][20].
- Approvals can be modeled using database buttons and automations (e.g., update status, notify approvers) [1][4].
- File sharing: supported types include HEIC, ICO, JPEG/JPG, PNG, TIF/TIFF, GIF, SVG, PDF, WEBP, MP3, MP4, WAV, OGG; secure URLs to uploads expire after 24 hours [21]. Free uploads up to 5 MB per file; Plus/Business/Enterprise allow unlimited file uploads [16].
- Communication: comments/discussions are native; for chat-like messaging teams typically integrate tools such as Slack via official integrations/embeds [18][7].
- Activity/audit: page and workspace analytics (viewers, views, editors); Enterprise plan includes Workspace Audit Log with event coverage and CSV export [22][23][24].
- External collaborators: Guests supported, alongside Members/Admins and Groups for permissioning [9].

### 3) Learning curve and ease of setup
- Official Guides hub and free Notion Academy courses (e.g., “How to build a connected workspace,” “Getting started with Projects”) [25][26][27].
- Solutions Partner directory for consulting implementations [28].
- Roles/groups/teamspaces documentation supports admins in configuring access and structure [9].

### 4) User interface design
- UI paradigm: page-based workspace where docs and databases coexist; databases have multiple views; left sidebar navigation; detailed “getting started” and “views” docs reflect this structure [29][2].
- Dark mode: available; appearance can follow system or be toggled; keyboard shortcut Cmd/Ctrl+Shift+L documented [30][31].
- Accessibility: release notes highlight accessibility considerations in dark mode palette updates [32].

### 5) Integration ecosystem and extensibility
- Identity/SSO/SCIM: official integration pages for Okta, Google Workspace, OneLogin, Rippling; SCIM provisioning docs; SAML SSO pages via those identity providers [33][34][35][36][42].
- Automation connectors and webhooks: Zapier integration is officially listed; RunReveal integration (audit logs webhook) is also listed [37][38].
- Public API: REST API with OAuth 2.0 for public integrations and tokens for internal apps; request limits guidance (~3 requests/second average per integration; 429 handling with Retry-After) [39][40][41].
- Integration terms clarify third-party status and limitations [46].
- Notion AI connectors (beta) for Slack and Google Drive documented with admin requirements [43][44].

### 6) U.S. pricing (free and paid)
- Plans and U.S. pricing: Free ($0), Plus ($10 per seat/month billed annually), Business ($15 per seat/month billed annually), Enterprise (Contact Sales) [16]. Notion announced Plus at $12 per member/month on monthly billing (June 2024) [45].
- Free plan limits: 10 guests; 7-day page history; 5 MB per-file upload; “Unlimited for individuals, limited block trial for 2+ members” [16].
- Key feature gating by tier (selected): Plus—unlimited blocks for teams, unlimited file uploads, 30-day history, 100 guests, synced databases, custom automations, charts/dashboards; Business—SAML SSO, private teamspaces, bulk PDF export, advanced page analytics, 90-day history, 250 guests; Enterprise—SCIM provisioning, advanced security/controls, audit log, workspace analytics, unlimited history [16].
- Trial: Plus Plan trial available “for up to 30 days” per Help Center [48].
- AI add-on: $8 per member/month annual, $10 monthly [16].
- Minimum seats: no minimums disclosed on the public pricing page for Plus/Business [16].

### 7) Material plan differences
- Page history: Free 7 days; Plus 30 days; Business 90 days; Enterprise unlimited [16].
- Guests: Free 10; Plus 100; Business 250; Enterprise 250+ [16].
- File uploads: Free 5 MB/file; paid plans unlimited [16].
- Enterprise-only controls: SCIM, audit log, advanced security & compliance integrations [16][24].
- Automations gating: database automations are paid; Free can create Slack notification automations only [4].
- Charts gating: Free can create one chart; paid plans unlimited [3].

## ClickUp

### 1) Customization capabilities and template flexibility
- Hierarchy and locations: Workspaces with Spaces, Folders, Lists (Free Forever limit of 5 Spaces; List/Folder per-Space limits vary by plan); List view is required; List and Board views are default; other views can be added (Calendar, Gantt, Timeline, Team, Workload, etc.) [46][47][48][49][50][51][52][53][54].
- Gantt/Timeline/Workload views: Dependencies and multi-scale Gantt (export to PDF/PNG on Business+); Timeline view grouping/time-range controls; Workload supports capacity-based scheduling and plan-based feature/usage limits [51][50][52][54].
- Custom fields: many types (Text, Long Text with 50,000 character limit, Dropdown/Labels, Money, Rating, Checkbox, Website, Date, Button, AI Fields, etc.); required fields on Business and above; workspace caps (e.g., 3,000 custom fields/workspace; options limits for labels) and “uses” limits on Free vs paid [55][56][57][58][59].
- Automations: available on all plans with monthly action limits by plan (e.g., Free: 5 active/100 actions; Unlimited: 500/1,000; Business: unlimited/5,000; Business Plus: unlimited/25,000; Enterprise: unlimited/250,000); triggers/actions lists published; templates available [61][62][63][64].
- Forms: can be created as views and via Forms Hub; plan-based capabilities include logic/options (Business Plus/Enterprise), authenticated forms (Enterprise), and export limits; Free includes 1 Form [65][66][67][68].
- Dashboards: plan-based usage caps (Free 60 uses; Unlimited 100; Business/Business Plus/Enterprise unlimited); card types by plan (e.g., Portfolio, bar/line charts, time-based analytics); dashboard and card-level filters [69][70][71][72][73][74][75].
- Permissions and roles: default permissions for guests/limited members/members/admins; custom roles (Business Plus: one custom role; Enterprise: unlimited) [76][77][78].
- Templates: create/share templates for nearly all objects (Spaces, Folders, Lists, Tasks, Docs, Views, Checklists, Whiteboards); Template Center for Workspace and community templates; directory categories list counts by category [79][80][81][82].

### 2) Collaboration features
- Real-time co-editing across Docs, task descriptions, Whiteboards; live cursors supported [83].
- Mentions and comments: @mentions across tasks/Docs/Chat; assigned comments create action items; flexible notification presets [84][85][86].
- Approvals/proofing: timesheet approvals feature; proofing/markup on images/videos/PDFs with plan-based usage limits (Free/Unlimited: 100 uses; Business+: unlimited) [87][88][89].
- File sharing limits: Free includes 100 MB total storage; max 1 GB per file; up to 1,000 attachments per task (limit documented Jan 2025) [90].
- Built-in chat: Channels and DMs are native; integrations include Google Chat for notifications [91][92].
- Activity feeds: Activity view at any location; task-level activity search/filtering [93][94].
- Guests/collaborators: default permissions and constraints documented; guests are view-only on Dashboards [76][95].

### 3) Learning curve and ease of setup
- Onboarding: “Set up your team’s Workspace” guidance; Getting Started collection; setup-from-scratch guidance (e.g., default List view) [96][97][98].
- Training: ClickUp University provides free courses, live training, certifications [99][100].
- Templates: official guidance on finding and applying templates [81].

### 4) User interface design
- Primary UI: hierarchy-driven with multiple views (List, Board, Gantt, Calendar, Timeline, Team, Workload); best practices documented [101].
- Dark mode and high-contrast mode available [102][103].
- Accessibility: ClickUp states a commitment toward WCAG compliance; VPAT available upon request via support [104].

### 5) Integration ecosystem and extensibility
- Integrations: vendor states “Connect over 1,000+ tools to ClickUp,” including native integrations, imports, and third-party connections [105].
- Automation connectors: Zapier and Make supported on all plans (role restrictions apply) [106][107].
- Public API: OAuth and token authentication; rate limits by plan (e.g., Free/Unlimited/Business: 100 requests/minute; Business Plus: 1,000 rpm; Enterprise: 10,000 rpm); API docs and usage guidance [108][109][110].
- Identity/SSO/SCIM: Google SSO (Business+); Microsoft, Okta, and custom SAML SSO (Enterprise); Okta SCIM provisioning guide; bypass SSO on Business+ and above [111][112][113][114][115][116].

### 6) U.S. pricing (free and paid)
- Plan names on public pricing page: Free Forever; Unlimited; Business; Enterprise (Contact Sales). USD prices are shown with a monthly/annual billing toggle; on the accessed version the page displayed $7/user/month (Unlimited) and $12/user/month (Business). Use the on-page toggle to confirm monthly vs annual in U.S. pricing at checkout [117]. Help Center enumerates plan names (including Business Plus) and defers to the pricing page for current pricing and currency [118].
- Trials and billing: upgrade paths and trial guidance are documented; Help Center notes trials but does not specify a fixed trial length for all plans; AI add-on trials are referenced in add-on FAQ [119][120].
- Free plan limits (selected): 100 MB total storage; max attachment size 1 GB; automations limit 5 active/100 actions per month; guests/limited members and integration usage are constrained by role/plan [90][61].

### 7) Material plan differences
- SSO and security: Google SSO available on Business+; Microsoft/Okta/custom SAML SSO on Enterprise; per-user SSO bypass on Business+ and above [111][116].
- Roles: custom roles available (Business Plus: one; Enterprise: unlimited) [76][78].
- Workload/Dashboards/Automations limits: Workload advanced features and usage caps vary by plan; Dashboards uses and card types gated by plan; automation monthly action limits increase by tier [54][69][70][61].
- Forms: logic/options/hidden fields on Business Plus/Enterprise; Account-Authenticated Forms on Enterprise; enabled via Security & Permissions [66][68][126].

## Practical selection notes (subjective synthesis; validate against cited features)
- Consider Notion when a team needs a flexible doc+database workspace with native forms and database-level automations, and wants to build lightweight dashboards inside the same pages; guardrails for structure can be set with granular database permissions; advanced audit and identity controls require Business/Enterprise [1][2][3][4][9][16][24][33][42].
- Consider ClickUp when a team wants a task-first hierarchy with many specialized views (Gantt, Timeline, Workload), built-in Dashboards with card types (including portfolio and time-based analytics), robust native automations at higher action volumes, and structured role/permission models including custom roles (Business+/Enterprise) and comprehensive SSO options (Enterprise) [51][52][54][69][70][71][74][75][61][76][78][111][114].

## Coverage gaps and next steps
- The research package did not include primary-source citations for Trello, Jira Software (Cloud), monday.com (Work Management), Asana, Microsoft Project (Project for the web), Smartsheet, or Airtable. To complete the brief across all seven dimensions for those products, gather official links to: pricing pages with U.S. billing toggles, plan/feature matrices, help center docs on custom fields/views/automations/forms/dashboards/permissions, collaboration docs (co-editing, comments, approvals, file caps), UI tours and accessibility/VPAT statements, integration directories/API docs/SSO guides, template galleries and counts, and trial terms.

### Sources
[1] Notion Help — Database properties: https://www.notion.so/en-us/help/database-properties  
[2] Notion Help — Views, filters & sorts: https://mobile.www.notion.so/help/views-filters-and-sorts  
[3] Notion Help — Chart view: https://www.notion.so/zh-tw/help/charts  
[4] Notion Help — Database automations: https://www.notion.so/help/database-automations?g-exp=decagon_launch--on  
[5] Notion Help — Database buttons: https://www.notion.so/zh-cn/help/database-buttons  
[6] Notion Help — Forms: https://www.notion.so/zh-cn/help/forms  
[7] Notion Help — Embeds, bookmarks, and link mentions: https://www.notion.so/help/embed-and-connect-other-apps  
[8] Notion Help — Link previews: https://mobile.www.notion.so/help/link-previews  
[9] Notion — Members, guests, and groups (roles): https://www.notion.so/f9a16ee752924ec1b6c6da328cf87e51  
[10] Notion Releases — 2022‑02‑08 (database permissions): https://www.notion.so/es/releases/2022-02-08  
[11] Notion Help — Assign custom database permissions: https://www.notion.so/es/help/guides/assign-custom-database-permissions  
[12] Notion Help — The ultimate guide to Notion templates: https://www.notion.so/help/guides/the-ultimate-guide-to-notion-templates  
[13] Notion Help — How to submit a template: https://www.notion.so/zh-cn/help/how-to-submit-a-template  
[14] Notion — Templates directory: https://www.notion.so/templates  
[15] Notion — Export to HTML (doc page): https://www.notion.so/HTML-ca4bbfb2a9a144d0a7f11586258b92e9  
[16] Notion — Pricing (US): https://www.notion.so/pricing  
[17] Notion Help — Collaborate within a workspace: https://mobile.www.notion.so/help/collaborate-within-a-workspace  
[18] Notion Help — Comments, mentions, and reminders: https://www.notion.so/de/help/comments-mentions-and-reminders  
[19] Notion — Mentions, discussions, comments (legacy): https://www.notion.so/Mentions-discussions-comments-d118e53008c14892908c6f48cfd3e944  
[20] Notion Help — Updates and notifications: https://www.notion.so/help/updates-and-notifications  
[21] Notion Help — I can’t upload or view a file: https://www.notion.so/es-es/help/cant-upload-or-view-a-file  
[22] Notion Help — Page analytics: https://www.notion.so/es-es/help/page-analytics  
[23] Notion Releases — 2023‑02‑09 (page & workspace analytics): https://www.notion.so/es/releases/2023-02-09  
[24] Notion Help — Audit log: https://www.notion.so/es/help/audit-log  
[25] Notion Help — Guides hub: https://www.notion.so/help/guides  
[26] Notion Academy — How to build a connected workspace: https://www.notion.so/es/help/notion-academy/course/how-to-build-a-connected-workspace  
[27] Notion Academy — Getting started with Projects: https://www.notion.so/es/help/notion-academy/course/getting-started-projects  
[28] Notion — Solutions Partners directory: https://www.notion.so/consulting-partners  
[29] Notion — Getting Started with Notion (official page): https://www.notion.so/notion/Getting-Started-with-Notion-f0e1a6d326d84d6984d948da96965045  
[30] Notion — Dark mode: https://www.notion.so/Dark-mode-2202b9c9e211446a947c1797c20a49e0  
[31] Notion Help — Appearance settings: https://www.notion.so/de/help/appearance-settings  
[32] Notion Releases — 2022‑03‑03 (dark mode palette/accessibility): https://www.notion.so/pt/releases/2022-03-03  
[33] Notion Integrations — Okta: https://www.notion.so/de/integrations/okta  
[34] Notion Integrations — Google Workspace: https://www.notion.so/pt/integrations/google-workspace  
[35] Notion Integrations — OneLogin: https://www.notion.so/integrations/onelogin  
[36] Notion Integrations — Rippling: https://www.notion.so/integrations/rippling  
[37] Notion Integrations — Zapier: https://www.notion.so/zh-cn/integrations/zapier  
[38] Notion Integrations — RunReveal: https://www.notion.so/integrations/runreveal  
[39] Notion Developers — API reference intro: https://developers.notion.com/reference/intro  
[40] Notion Developers — Authorization: https://developers.notion.com/docs/authorization  
[41] Notion Developers — Request limits: https://developers.notion.com/reference/request-limits  
[42] Notion Help — Provision users & groups with SCIM: https://www.notion.so/zh-cn/help/provision-users-and-groups-with-scim  
[43] Notion — AI connectors (Slack/Google): https://www.notion.so/zh-cn/ai-connectors  
[44] Notion Help — Notion AI connectors for Slack: https://www.notion.so/es/help/notion-ai-connectors-for-slack  
[45] Notion Releases — 2024‑06‑26 (Plus plan pricing/monthly): https://www.notion.so/zh-tw/releases/2024-06-26  
[46] ClickUp Help — Intro to Spaces: https://help.clickup.com/hc/en-us/articles/6309466958103-Intro-to-Spaces  
[47] ClickUp Help — Intro to Folders: https://help.clickup.com/hc/en-us/articles/6311450560407-Intro-to-Folders  
[48] ClickUp Help — Add a view: https://help.clickup.com/hc/en-us/articles/26032576190615-Add-a-view  
[49] ClickUp Help — Make a required view: https://help.clickup.com/hc/en-us/articles/6310293474199-Make-a-required-view  
[50] ClickUp Help — Add a Timeline view: https://help.clickup.com/hc/en-us/articles/6310399909143-Add-a-Timeline-view  
[51] ClickUp Help — Gantt View: https://help.clickup.com/hc/en-us/articles/6310249474967-Gantt-View  
[52] ClickUp Help — Use Workload view: https://help.clickup.com/hc/en-us/articles/6310449699735-Use-Workload-view  
[53] ClickUp Help — Set the time period in Workload view: https://help.clickup.com/hc/en-us/articles/30799743741847-Set-the-time-period-in-Workload-view  
[54] ClickUp Help — Workload view feature availability and limits: https://help.clickup.com/hc/en-us/articles/30657456679703-Workload-view-feature-availability-and-limits  
[55] ClickUp Help — Custom Field Types: https://help.clickup.com/hc/en-us/articles/6303499162647-Custom-Field-Types  
[56] ClickUp Help — Make Custom Fields required: https://help.clickup.com/hc/en-us/articles/30407234676887-Make-Custom-Fields-required  
[57] ClickUp Help — Intro to Custom Field Manager: https://help.clickup.com/hc/en-us/articles/13066263096727-Intro-to-Custom-Field-Manager  
[58] ClickUp Help — Custom Fields uses: https://help.clickup.com/hc/en-us/articles/10993484102167-Custom-Fields-uses  
[59] ClickUp Help — Create a Custom Field with Custom Field Manager: https://help.clickup.com/hc/en-us/articles/14276808327063-Create-a-Custom-Field-with-Custom-Field-Manager  
[60] ClickUp Help — Set default values for Custom Fields: https://help.clickup.com/hc/en-us/articles/30406587945879-Set-default-values-for-Custom-Fields  
[61] ClickUp Help — Automations feature availability and limits: https://help.clickup.com/hc/en-us/articles/23477062949911-Automations-feature-availability-and-limits  
[62] ClickUp Help — Use Automation Triggers: https://help.clickup.com/hc/en-us/articles/6312128853015-Use-Automation-Triggers  
[63] ClickUp Help — Use Automation Actions: https://help.clickup.com/hc/en-us/articles/6312097314199-Use-Automation-Actions  
[64] ClickUp Help — Use Automations templates: https://help.clickup.com/hc/en-us/articles/6312150815255-Use-Automations-templates  
[65] ClickUp Help — Create and access Forms: https://help.clickup.com/hc/en-us/articles/30687934700055-Create-and-access-Forms  
[66] ClickUp Help — Form view feature availability and limits: https://help.clickup.com/hc/en-us/articles/25810829634711-Form-view-feature-availability-and-limits  
[67] ClickUp Help — Forms Hub: https://help.clickup.com/hc/en-us/articles/26301017413911-Forms-Hub  
[68] ClickUp Help — Account-Authenticated Forms: https://help.clickup.com/hc/en-us/articles/33384203086359-Account-Authenticated-Forms  
[69] ClickUp Help — Dashboards feature availability and limits: https://help.clickup.com/hc/en-us/articles/21257864098071-Dashboards-feature-availability-and-limits  
[70] ClickUp Help — Cards feature availability and limits: https://help.clickup.com/hc/en-us/articles/26985557505303-Cards-feature-availability-and-limits  
[71] ClickUp Help — Portfolio cards: https://help.clickup.com/hc/en-us/articles/6312200675991-Portfolio-cards  
[72] ClickUp Help — Bar Chart cards: https://help.clickup.com/hc/en-us/articles/14670260184855-Bar-Chart-cards  
[73] ClickUp Help — Line Chart cards: https://help.clickup.com/hc/en-us/articles/6312238252311-Line-Chart-cards  
[74] ClickUp Help — Use time-based Dashboard cards: https://help.clickup.com/hc/en-us/articles/16956329804311-Use-time-based-Dashboard-cards  
[75] ClickUp Help — Use Dashboard filters: https://help.clickup.com/hc/en-us/articles/6312203092119-Use-Dashboard-filters  
[76] ClickUp Help — Default permissions for guests/limited members/members/admins: https://help.clickup.com/hc/en-us/articles/26603207201431-Default-permissions-for-guests-limited-members-members-and-admins  
[77] ClickUp Help — Permissions in detail: https://help.clickup.com/hc/en-us/articles/6309221065495-Permissions-in-detail  
[78] ClickUp Help — Custom role permissions: https://help.clickup.com/hc/en-us/articles/6310020641175-Custom-role-permissions  
[79] ClickUp Help — Create a template: https://help.clickup.com/hc/en-us/articles/6326066114455-Create-a-template  
[80] ClickUp Help — Share a template: https://help.clickup.com/hc/en-us/articles/6326130026519-Share-a-template  
[81] ClickUp Help — Find a template: https://help.clickup.com/hc/en-us/articles/6326080034199-Find-a-template  
[82] ClickUp — Templates categories directory: https://clickup.com/templates/categories  
[83] ClickUp Help — Edit collaboratively: https://help.clickup.com/hc/en-us/articles/6311275622423-Edit-collaboratively  
[84] ClickUp Help — Use mentions: https://help.clickup.com/hc/en-us/articles/6311550474263-Use-mentions  
[85] ClickUp Help — Comments overview: https://help.clickup.com/hc/en-us/articles/6309646134295-Comments-overview  
[86] ClickUp Help — Notification settings: https://help.clickup.com/hc/en-us/articles/6325918957335-Notification-settings  
[87] ClickUp Help — Timesheet approvals: https://help.clickup.com/hc/en-us/articles/27090646766231-Timesheet-approvals  
[88] ClickUp Help (PT-BR) — Design reviews: https://help.clickup.com/hc/pt-br/articles/6328033402007-Use-o-ClickUp-para-revis%C3%B5es-de-design  
[89] ClickUp Help — Annotate images, videos, and PDFs with Proofing: https://help.clickup.com/hc/en-us/articles/6325985679383-Annotate-images-videos-and-PDFs-with-Proofing  
[90] ClickUp Help — Attachments in Tasks: https://help.clickup.com/hc/en-us/articles/6309666546199-Attachments-in-Tasks  
[91] ClickUp Help — What is Chat: https://help.clickup.com/hc/en-us/articles/25790737416855-What-is-Chat  
[92] ClickUp Help — Google Chat integration: https://help.clickup.com/hc/en-us/articles/6305822807703-Google-Chat-integration  
[93] ClickUp Help — Intro to Activity view: https://help.clickup.com/hc/en-us/articles/6310079875095-Intro-to-Activity-view  
[94] ClickUp Help — Search and filter task activity: https://help.clickup.com/hc/en-us/articles/6309921223191-Search-and-filter-task-activity  
[95] ClickUp Help — Dashboards Hub: https://help.clickup.com/hc/en-us/articles/14236332445335-Dashboards-Hub  
[96] ClickUp Help — Set up your team’s Workspace: https://help.clickup.com/hc/en-us/articles/9563779819031-Set-up-your-team-s-Workspace  
[97] ClickUp Help — Getting Started category: https://help.clickup.com/hc/en-us/categories/5735245366551-Getting-Started  
[98] ClickUp Help — Set up your team’s Workspace from scratch: https://help.clickup.com/hc/en-us/articles/10636005013271-Set-up-your-team-s-Workspace-from-scratch  
[99] ClickUp Help (IT) — Introduzione a ClickUp University: https://help.clickup.com/hc/it/articles/9865266661655-Introduzione-a-ClickUp-University  
[100] ClickUp Help — Intro to ClickUp University: https://help.clickup.com/hc/en-us/articles/9865266661655-Intro-to-ClickUp-University  
[101] ClickUp Help — Hierarchy best practices: https://help.clickup.com/hc/en-us/articles/20480724378135-Hierarchy-best-practices  
[102] ClickUp Help — Dark Mode: https://help.clickup.com/hc/en-us/articles/6309134405655-Dark-Mode  
[103] ClickUp Help — Use High Contrast Mode: https://help.clickup.com/hc/en-us/articles/6310793610391-Use-High-Contrast-Mode  
[104] ClickUp — Accessibility: https://clickup.com/accessibility  
[105] ClickUp — Integrations overview: https://clickup.com/integrations  
[106] ClickUp Help — Zapier integration: https://help.clickup.com/hc/en-us/articles/6306186099223-Zapier-integration  
[107] ClickUp Help (ES) — Make integration: https://help.clickup.com/hc/es/articles/6305664103959-Crea-integraciones-personalizadas-con-Make  
[108] ClickUp Developer Docs — Index: https://developer.clickup.com/docs/index  
[109] ClickUp Developer Docs — Rate limits: https://developer.clickup.com/docs/rate-limits  
[110] ClickUp Help — Use the ClickUp API: https://help.clickup.com/hc/en-us/articles/6303426241687-Use-the-ClickUp-API  
[111] ClickUp Help — Intro to single sign-on (SSO): https://help.clickup.com/hc/en-us/articles/6305043992343-Intro-to-single-sign-on-SSO  
[112] ClickUp Help — Google SSO: https://help.clickup.com/hc/en-us/articles/6305105918615-Google-Single-Sign-On  
[113] ClickUp Help — Microsoft SSO: https://help.clickup.com/hc/en-us/articles/6305100485015-Microsoft-single-sign-on  
[114] ClickUp Help — Configure custom SAML SSO: https://help.clickup.com/hc/en-us/articles/6305027039767-Configure-custom-SAML-single-sign-on  
[115] ClickUp Help — Okta SCIM configuration guide: https://help.clickup.com/hc/en-us/articles/6305052795287-Okta-SCIM-ClickUp-configuration-guide  
[116] ClickUp Help — Bypass SSO for individual users: https://help.clickup.com/hc/en-us/articles/28428003934743-Bypass-single-sign-on-SSO-for-individual-users  
[117] ClickUp — Pricing page (US localized view with toggle): https://clickup.com/de/pricing  
[118] ClickUp Help — Intro to pricing: https://help.clickup.com/hc/en-us/articles/10129535087383-Intro-to-pricing  
[119] ClickUp Help — Upgrade your plan: https://help.clickup.com/hc/en-us/articles/6303314345623-Upgrade-your-plan  
[120] ClickUp Help — Add-on FAQ (trials): https://help.clickup.com/hc/en-us/articles/6303101719831-Add-on-FAQ  
[121] ClickUp Help (ES) — Embed content in ClickUp: https://help.clickup.com/hc/es/articles/6305793042455-Incorporar-contenido-en-ClickUp  
[122] ClickUp Help — Intro to Subtasks: https://help.clickup.com/hc/en-us/articles/6309825777943-Intro-to-Subtasks  
[123] ClickUp Help — Create tasks and subtasks on mobile: https://help.clickup.com/hc/en-us/articles/15147770180759-Create-tasks-and-subtasks-on-mobile  
[124] ClickUp Help — Create and share a Dashboard view: https://help.clickup.com/hc/en-us/articles/21501348696727-Create-and-share-a-Dashboard-view  
[125] ClickUp Help — ClickUp Support: https://help.clickup.com/hc/en-us/articles/16251448728727-ClickUp-Support  
[126] ClickUp Help — Configure advanced Workspace security permissions: https://help.clickup.com/hc/en-us/articles/6309213140375-Configure-advanced-Workspace-security-permissions