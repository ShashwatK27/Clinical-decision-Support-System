import { useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Html } from "@react-three/drei";
import * as THREE from "three";

interface ScannerSceneProps {
  isScanning: boolean;
  scanProgress: number;
}

// Particle stream that emanates from the sweeping laser
function ScanParticles({ isScanning, laserY }: { isScanning: boolean; laserY: number }) {
  const pointsRef = useRef<THREE.Points>(null);
  const particleCount = 200;

  // Pre-fill coordinate arrays
  const positions = new Float32Array(particleCount * 3);
  const velocities = new Float32Array(particleCount * 3);
  
  for (let i = 0; i < particleCount; i++) {
    // Distribute around the center laser line width
    positions[i * 3] = (Math.random() - 0.5) * 2.8; 
    positions[i * 3 + 1] = laserY; // Start at laser height
    positions[i * 3 + 2] = 0.05 + Math.random() * 0.1;
    
    velocities[i * 3] = (Math.random() - 0.5) * 0.5;
    velocities[i * 3 + 1] = -Math.random() * 0.8 - 0.2; // Fall down
    velocities[i * 3 + 2] = (Math.random() - 0.5) * 0.3;
  }

  useFrame((state, delta) => {
    if (!pointsRef.current) return;
    const geo = pointsRef.current.geometry;
    const pos = geo.attributes.position;
    
    for (let i = 0; i < particleCount; i++) {
      if (!isScanning) {
        // Fade out or group at zero
        pos.setY(i, -999);
        continue;
      }
      
      // Update Y position (downwards stream)
      let y = pos.getY(i) + velocities[i * 3 + 1] * delta;
      let x = pos.getX(i) + velocities[i * 3] * delta;
      let z = pos.getZ(i) + velocities[i * 3 + 2] * delta;
      
      // Reset particle when it goes too far down or fades
      if (y < laserY - 0.8) {
        x = (Math.random() - 0.5) * 2.6;
        y = laserY + (Math.random() - 0.5) * 0.05; // Spurt from laser plane
        z = 0.08 + Math.random() * 0.05;
      }
      
      pos.setX(i, x);
      pos.setY(i, y);
      pos.setZ(i, z);
    }
    
    pos.needsUpdate = true;
  });

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          args={[positions, 3]}
        />
      </bufferGeometry>
      <pointsMaterial
        color="#06b6d4"
        size={0.038}
        transparent
        opacity={0.85}
        blending={THREE.AdditiveBlending}
        depthWrite={false}
      />
    </points>
  );
}

// Hovering Prescription Card
function PrescriptionCard({ isScanning, laserRef }: { isScanning: boolean; laserRef: React.RefObject<THREE.Mesh> }) {
  const cardRef = useRef<THREE.Group>(null);

  useFrame((state) => {
    const t = state.clock.getElapsedTime();
    
    // Smooth floating animation
    if (cardRef.current) {
      cardRef.current.position.y = Math.sin(t * 1.5) * 0.08 + 0.15;
      cardRef.current.rotation.y = Math.cos(t * 0.8) * 0.04;
      cardRef.current.rotation.x = Math.sin(t * 0.5) * 0.02 + 0.05;
    }
    
    // Sweep the laser plane horizontally/vertically over the card
    if (laserRef.current) {
      if (isScanning) {
        // Ping-pong sweep between Y = -0.7 and Y = 0.7
        laserRef.current.position.y = Math.sin(t * 3.5) * 0.7;
        laserRef.current.visible = true;
      } else {
        laserRef.current.visible = false;
      }
    }
  });

  return (
    <group ref={cardRef}>
      {/* Volumetric Card Mesh */}
      <mesh castShadow receiveShadow>
        <boxGeometry args={[2.7, 1.8, 0.05]} />
        <meshPhysicalMaterial 
          color="#0a1226" 
          roughness={0.15} 
          metalness={0.1}
          clearcoat={1.0}
          clearcoatRoughness={0.1}
          transmission={0.4} // Transparent glass card feel
          thickness={0.2}
          transparent
          opacity={0.9}
        />
      </mesh>

      {/* Cybernetic Grid Frame Overlay */}
      <mesh position={[0, 0, 0.026]}>
        <planeGeometry args={[2.65, 1.75]} />
        <meshBasicMaterial 
          color="#3b82f6" 
          wireframe 
          transparent 
          opacity={0.15} 
        />
      </mesh>

      {/* Futuristic Medical Prescription HUD Content (rendered as a crisp high-DPI HUD overlay) */}
      <Html position={[0, 0, 0.04]} center>
        <div style={{
          width: 240,
          height: 156,
          padding: "10px 12px",
          background: "rgba(10, 20, 42, 0.92)",
          border: "1px solid rgba(6, 182, 212, 0.5)",
          borderRadius: 12,
          color: "#f8fafc",
          fontFamily: "'Outfit', monospace",
          fontSize: 10,
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-between",
          boxShadow: "inset 0 0 15px rgba(6, 182, 212, 0.2), 0 20px 50px rgba(0,0,0,0.7)",
          pointerEvents: "none",
          userSelect: "none"
        }}>
          {/* Header */}
          <div style={{ display: "flex", justifyContent: "space-between", borderBottom: "1px solid rgba(6, 182, 212, 0.3)", paddingBottom: 3 }}>
            <span style={{ fontWeight: 800, color: "#06b6d4", fontSize: 9, letterSpacing: 0.8 }}>CDSS RX INTAKE v3.0</span>
            <span style={{ color: "#10b981", fontWeight: 700, fontSize: 8 }}>● ONLINE</span>
          </div>

          {/* Clinician & Patient Mock Details */}
          <div style={{ display: "flex", flexDirection: "column", gap: 2, margin: "4px 0", color: "rgba(248, 250, 252, 0.85)", fontSize: 9 }}>
            <div><strong style={{ color: "#3b82f6" }}>PATIENT:</strong> John Doe (CLN-70)</div>
            <div><strong style={{ color: "#3b82f6" }}>PRESCRIBER:</strong> DR. H. SHAW</div>
          </div>

          {/* OCR Box Grid */}
          <div style={{ 
            background: "rgba(5, 10, 20, 0.85)", 
            padding: "4px 6px", 
            borderRadius: 6, 
            border: "1px dashed rgba(6, 182, 212, 0.3)",
            fontSize: 8.5,
            color: "#38bdf8",
            display: "flex",
            flexDirection: "column",
            gap: 1
          }}>
            <div>▶ Active Regimen:</div>
            <div style={{ color: "#f8fafc", fontStyle: "italic", fontSize: 8 }}>
              warfarin + ibuprofen + amoxicillin + loratadine
            </div>
          </div>

          {/* Footer Signature Bar */}
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: 7, color: "rgba(255,255,255,0.4)", marginTop: 2 }}>
            <span>HASH: CD80-F5E2</span>
            <span style={{ fontStyle: "italic", color: "#06b6d4" }}>Shaw verified</span>
          </div>
        </div>
      </Html>
    </group>
  );
}

// Flat Glass Holographic Scanning Bed Plate
function GlassScanningPlate() {
  return (
    <group position={[0, -0.9, 0]} rotation={[-Math.PI / 2.05, 0, 0]}>
      {/* Symmetrical Outer Metal Rim */}
      <mesh receiveShadow>
        <boxGeometry args={[4.4, 3.4, 0.1]} />
        <meshStandardMaterial color="#0f172a" metalness={0.9} roughness={0.3} />
      </mesh>

      {/* Inner Glowing Plate grid */}
      <mesh position={[0, 0, 0.055]}>
        <planeGeometry args={[4.1, 3.1]} />
        <meshBasicMaterial color="#06b6d4" wireframe transparent opacity={0.15} />
      </mesh>
      
      {/* Cyan neon glowing border frame */}
      <mesh position={[0, 0, 0.06]}>
        <planeGeometry args={[4.05, 3.05]} />
        <meshPhysicalMaterial 
          color="#0284c7" 
          transparent
          opacity={0.3}
          roughness={0.1}
          transmission={0.95} 
          thickness={0.1}
        />
      </mesh>
    </group>
  );
}

export default function ScannerScene({ isScanning, scanProgress }: ScannerSceneProps) {
  const laserRef = useRef<THREE.Mesh>(null);

  return (
    <div style={{ width: "100%", height: "100%", minHeight: 400, position: "relative" }}>
      <Canvas 
        camera={{ position: [0, 0.3, 4.4], fov: 42 }} 
        shadows
        gl={{ antialias: true, alpha: true }}
      >
        <ambientLight intensity={0.25} />
        
        {/* Soft volumetric spotlight focusing on prescription */}
        <spotLight 
          position={[0, 4, 3]} 
          angle={0.4} 
          penumbra={0.8} 
          intensity={1.8} 
          castShadow 
          shadow-mapSize-width={1024} 
          shadow-mapSize-height={1024}
        />

        {/* Ambient backlighting to bring out depth */}
        <directionalLight position={[0, -2, -3]} intensity={0.4} color="#3b82f6" />
        
        <GlassScanningPlate />
        
        <PrescriptionCard isScanning={isScanning} laserRef={laserRef} />

        {/* Sweeping Laser Line Mesh */}
        <mesh ref={laserRef} position={[0, 0, 0.065]} castShadow>
          <boxGeometry args={[2.85, 0.04, 0.06]} />
          <meshBasicMaterial color="#00f3ff" />
        </mesh>
        
        {/* Particle sparks triggered during scanner operations */}
        <ScanParticles isScanning={isScanning} laserY={laserRef.current ? laserRef.current.position.y : 0} />

        <OrbitControls 
          enableZoom={false} 
          maxPolarAngle={Math.PI / 1.8} 
          minPolarAngle={Math.PI / 2.3}
          maxAzimuthAngle={Math.PI / 6}
          minAzimuthAngle={-Math.PI / 6}
        />
      </Canvas>
    </div>
  );
}
