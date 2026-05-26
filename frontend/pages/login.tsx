import { useState, type FormEvent } from "react";
import axios from "axios";
import Link from "next/link";
import NavBar from "../components/NavBar";

export default function LoginPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [message, setMessage] = useState<string | null>(null);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    try {
      const response = await axios.post("http://localhost:8000/api/token", new URLSearchParams({ username: email, password }));
      setMessage("Logged in successfully. Token stored in memory for demo.");
      console.log("Access token:", response.data.access_token);
    } catch (error) {
      setMessage("Login failed. Check credentials and backend availability.");
    }
  };

  return (
    <div>
      <NavBar />
      <main className="container" style={{ paddingTop: 40, paddingBottom: 40 }}>
        <section className="card" style={{ maxWidth: 520, margin: "0 auto" }}>
          <h1 style={{ marginBottom: 12 }}>Sign in to MediIntel</h1>
          <p className="lead">Your secure clinician account to manage prescriptions, history, and AI insights.</p>
          <form onSubmit={handleSubmit}>
            <input
              className="input"
              placeholder="Email address"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
            <input
              className="input"
              placeholder="Password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
            <button className="button" type="submit">
              Sign in
            </button>
          </form>
          {message ? <p style={{ marginTop: 18 }}>{message}</p> : null}
          <p style={{ marginTop: 24 }}>
            New user? <Link href="/register">Create an account</Link>
          </p>
        </section>
      </main>
    </div>
  );
}
