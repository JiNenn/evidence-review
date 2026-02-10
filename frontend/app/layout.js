import "./globals.css";

export const metadata = {
  title: "diffUI",
  description: "Evidence-first comparison UI"
};

export default function RootLayout({ children }) {
  return (
    <html lang="ja">
      <body>{children}</body>
    </html>
  );
}
