# Research Paper Answer Bot: web app

Next.js (App Router, TypeScript, Tailwind CSS) front end over the FastAPI service in `backend/api`.
The layout follows the design mockup: chats and settings on the left, the conversation in the
middle, and a reader on the right that shows a cited passage on its page or the paper library.

## Run

Start Ollama and the API first (from `backend/`: `.venv\Scripts\uvicorn api.main:app --port 8000`), then:

```powershell
npm install
npm run dev        # http://localhost:3000, or: npm run build; npm start
```

The app calls the API at `http://localhost:8000`. To use another address, set
`NEXT_PUBLIC_API_URL` when building or starting it.

## Layout

| Path | Contents |
|---|---|
| `app/layout.tsx` | Fonts (IBM Plex Sans/Mono, Source Serif 4) and the saved-theme script |
| `app/globals.css` | Design tokens for light and dark, mapped into Tailwind's theme, and the component styles |
| `components/AnswerBot.tsx` | State, health polling, the streaming ask flow, sessions, theme |
| `components/Turn.tsx` | One question and answer: steps, source cards, citations, refusal, error, trace |
| `components/Reader.tsx` | Passage view with the cited sentence highlighted, and the paper library filter |
| `lib/types.ts` | TypeScript mirrors of the API's Pydantic models |
| `lib/api.ts` | API client, including the Server-Sent Events parser for `POST /ask/stream` |

The app renders in the browser only (`components/ClientApp.tsx`), because it starts from this
tab's session id and the saved theme, which live in browser storage.
