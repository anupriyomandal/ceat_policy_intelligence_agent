import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "CEAT Policy Intelligence Agent",
  description: "Policy Q&A grounded in CEAT policy documents.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
