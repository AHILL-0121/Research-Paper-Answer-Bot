"use client";

import dynamic from "next/dynamic";

// The app reads this tab's session and the saved theme from browser storage as it starts,
// so it renders in the browser only; the server sends the desk background and fonts.
const AnswerBot = dynamic(() => import("./AnswerBot"), {
  ssr: false,
  loading: () => <div className="boot" aria-busy="true" />,
});

export default function ClientApp() {
  return <AnswerBot />;
}
