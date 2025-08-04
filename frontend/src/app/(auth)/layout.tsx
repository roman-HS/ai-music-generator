import "~/styles/globals.css";

import { Geist } from "next/font/google";
import { Providers } from "~/components/providers";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Music Generator",
  description: "Music Generator",
  icons: [{ rel: "icon", url: "/favicon.ico" }],
};

const geist = Geist({
  subsets: ["latin"],
  variable: "--font-geist-sans",
});

export default function AuthLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${geist.variable}`}>
      <body className="flex min-h-screen flex-col">
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
