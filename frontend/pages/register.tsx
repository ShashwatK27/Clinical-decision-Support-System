import { useState, type FormEvent } from "react";
import axios from "axios";
import Link from "next/link";
import NavBar from "../components/NavBar";

export default function RegisterPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [fullName, setFullName] = useState("");
  const [message, setMessage] = useState<string | null>(null);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    try {
      await axios.post("http://localhost:8000/api/register", { email, password, full_name: fullName });
      setMessage("Account created successfully. Please sign in.");
    } catch (error) {
      setMessage("Registration failed. Check the backend or try a different email.");
    }
  };

  return (
    <div>
      <NavBar />
      <main className="container" style={{ paddingTop: 40, paddingBottom: 40 }}>
        <section className="card" style={{ maxWidth: 520, margin: "0 auto" }}>
          <h1 style={{ marginBottom: 12 }}>Create your account</h1>
          <p className="lead">Start building your personal prescription history and analysis dashboard.</p>
          <form onSubmit={handleSubmit}>
            <input
              className="input"
              placeholder="Full name"
              value={fullName}
              onChange={(e) => setFullName(e.target.value)}
            />
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
              Register
            </button>
          </form>
          {message ? <p style={{ marginTop: 18 }}>{message}</p> : null}
          <p style={{ marginTop: 24 }}>
            Already have an account? <Link href="/login">Sign in</Link>
          </p>
        </section>
      </main>
    </div>
  );
}
