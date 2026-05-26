import { Canvas } from "@react-three/fiber";
import { Float, OrbitControls, Html, Environment } from "@react-three/drei";

function PrescriptionCard() {
  return (
    <Float rotationIntensity={0.2} floatIntensity={1.2} speed={1.2}>
      <mesh position={[0, 0, 0]} castShadow receiveShadow>
        <boxGeometry args={[2.7, 1.6, 0.14]} />
        <meshStandardMaterial color="#2563eb" metalness={0.3} roughness={0.2} />
      </mesh>
      <Html position={[0, 0, 0.09]} center>
        <div style={{ width: 260, padding: 18, borderRadius: 20, background: "rgba(255,255,255,0.92)", boxShadow: "0 30px 90px rgba(15, 23, 42, 0.18)", color: "#0f172a" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
            <span style={{ fontWeight: 700, fontSize: 14, color: "#1d4ed8" }}>MEDIINTEL RX</span>
            <span style={{ width: 12, height: 12, borderRadius: "50%", background: "#34d399" }} />
          </div>
          <div style={{ marginBottom: 14 }}>
            <div style={{ height: 12, width: "78%", background: "#e2e8f0", borderRadius: 9999, marginBottom: 10 }} />
            <div style={{ height: 12, width: "62%", background: "#e2e8f0", borderRadius: 9999 }} />
          </div>
          <div style={{ display: "grid", gap: 8 }}>
            <div style={{ height: 10, width: "100%", background: "#dbeafe", borderRadius: 9999 }} />
            <div style={{ height: 10, width: "85%", background: "#dbeafe", borderRadius: 9999 }} />
            <div style={{ height: 10, width: "92%", background: "#dbeafe", borderRadius: 9999 }} />
          </div>
        </div>
      </Html>
    </Float>
  );
}

function FloatingBadge({ position, title }: { position: [number, number, number]; title: string }) {
  return (
    <Float rotationIntensity={0.5} floatIntensity={2} speed={1.1}>
      <group position={position}>
        <mesh>
          <sphereGeometry args={[0.22, 32, 32]} />
          <meshStandardMaterial color="#f59e0b" emissive="#fbbf24" emissiveIntensity={0.3} roughness={0.35} />
        </mesh>
        <Html center position={[0, 0.4, 0]}>
          <div style={{ padding: "8px 12px", background: "rgba(15,23,42,0.88)", color: "white", borderRadius: 9999, fontSize: 12, whiteSpace: "nowrap" }}>{title}</div>
        </Html>
      </group>
    </Float>
  );
}

export default function HeroScene() {
  return (
    <div style={{ width: "100%", height: "100%", minHeight: 460 }}>
      <Canvas camera={{ position: [0, 0, 7], fov: 38 }} shadows>
        <ambientLight intensity={0.5} />
        <directionalLight position={[4, 6, 5]} intensity={1.2} castShadow shadow-mapSize-width={1024} shadow-mapSize-height={1024} />
        <Environment preset="city" />
        <PrescriptionCard />
        <FloatingBadge position={[-2.1, 1.1, 0]} title="OCR Ready" />
        <FloatingBadge position={[1.8, -0.9, 0]} title="Smart Alerts" />
        <OrbitControls enableZoom={false} autoRotate autoRotateSpeed={0.8} />
      </Canvas>
    </div>
  );
}
