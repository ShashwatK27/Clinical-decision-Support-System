import type { NextApiRequest, NextApiResponse } from "next";

// Simple in-memory token bucket rate limiter to prevent denial of service
const rateLimitCache = new Map<string, { tokens: number; lastRefill: number }>();
const LIMIT_WINDOW = 60000; // 1 minute
const MAX_TOKENS = 3;       // 3 submissions per minute

function isRateLimited(ip: string): boolean {
  const now = Date.now();
  const userData = rateLimitCache.get(ip) || { tokens: MAX_TOKENS, lastRefill: now };

  // Calculate token refill based on elapsed time
  const elapsed = now - userData.lastRefill;
  if (elapsed > LIMIT_WINDOW) {
    userData.tokens = MAX_TOKENS;
    userData.lastRefill = now;
  }

  if (userData.tokens <= 0) {
    return true;
  }

  userData.tokens -= 1;
  rateLimitCache.set(ip, userData);
  return false;
}

// Simple email regex validation
const EMAIL_REGEX = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

function sanitizeInput(val: string): string {
  if (typeof val !== "string") return "";
  return val
    .trim()
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#x27;")
    .replace(/\//g, "&#x2F;");
}

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  // 1. Enforce strict POST method filter
  if (req.method !== "POST") {
    res.setHeader("Allow", ["POST"]);
    return res.status(405).json({ error: `Method ${req.method} Not Allowed` });
  }

  try {
    // 2. Client IP detection
    const clientIp = (req.headers["x-forwarded-for"] as string) || req.socket.remoteAddress || "anonymous";

    // 3. Evaluate Throttling
    if (isRateLimited(clientIp)) {
      return res.status(429).json({ error: "Too many requests. Please wait a minute before resubmitting." });
    }

    // 4. Validate Payload Size Limit (Max 10KB to avoid memory attacks)
    const payloadLength = parseInt(req.headers["content-length"] || "0", 10);
    if (payloadLength > 10240) {
      return res.status(400).json({ error: "Payload exceeds safe bounds (Max 10KB)." });
    }

    const { name, email, subject, message } = req.body;

    // 5. Input Validations
    if (!name || !email || !message) {
      return res.status(400).json({ error: "Missing required values: name, email, and message." });
    }

    if (!EMAIL_REGEX.test(email)) {
      return res.status(400).json({ error: "Invalid email format." });
    }

    // 6. Sanitization
    const cleanName = sanitizeInput(name);
    const cleanEmail = email.trim().toLowerCase();
    const cleanSubject = sanitizeInput(subject || "MediSensonic Corporate Inquiry");
    const cleanMessage = sanitizeInput(message);

    // =========================================================================
    // 🗄️ DATABASE CONNECTION HOOK (Future PostgreSQL/SQLite CDSS database)
    // =========================================================================
    // Example:
    // await db.contactSubmissions.create({
    //   data: {
    //     name: cleanName,
    //     email: cleanEmail,
    //     subject: cleanSubject,
    //     message: cleanMessage,
    //     ip: clientIp,
    //     createdAt: new Date()
    //   }
    // });
    // =========================================================================

    // Keep console logs clear to meet production rules (avoiding standard logs)
    return res.status(200).json({ success: true, message: "Inquiry secured successfully." });

  } catch (error) {
    return res.status(500).json({ error: "Internal server error. Submissions could not be recorded." });
  }
}
