import { Canvas, useFrame } from "@react-three/fiber";
import { useRef, useMemo } from "react";
import * as THREE from "three";

// 1. Drifting Molecular Particle Node Field
function FloatingNetwork() {
  const pointsRef = useRef<THREE.Points>(null);
  const count = 100; // Constrain count to ensure absolute fluid 60 FPS on mobile

  const [positions] = useMemo(() => {
    const pos = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      // Distribute in a spherical volume
      const r = 2 + Math.random() * 8;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      
      pos[i * 3] = r * Math.sin(phi) * Math.cos(theta);
      pos[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      pos[i * 3 + 2] = r * Math.cos(phi) - 2; // Offset slightly deeper
    }
    return [pos];
  }, []);

  useFrame((state, delta) => {
    if (!pointsRef.current) return;
    // Slow ambient rotation drift of particles
    pointsRef.current.rotation.y += 0.015 * delta;
    pointsRef.current.rotation.x += 0.008 * delta;

    // Soft breathing scale animation
    const t = state.clock.getElapsedTime();
    const breathe = 1.0 + Math.sin(t * 0.3) * 0.04;
    pointsRef.current.scale.set(breathe, breathe, breathe);
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
        color="#2FD8D5" // Medical Cyan glow
        size={0.06}
        transparent
        opacity={0.15} // Extremely faint and ambient
        depthWrite={false}
        blending={THREE.AdditiveBlending}
      />
    </points>
  );
}

// 2. Holographic Wireframe Geometric Core
function HolographicCore() {
  const meshRef = useRef<THREE.Mesh>(null);

  useFrame((state, delta) => {
    if (!meshRef.current) return;
    // Slow rotational drift
    meshRef.current.rotation.y += 0.008 * delta;
    meshRef.current.rotation.z += 0.004 * delta;

    // Slow ambient vertical floating sway
    const t = state.clock.getElapsedTime();
    meshRef.current.position.y = Math.sin(t * 0.35) * 0.2;
  });

  return (
    <mesh ref={meshRef} position={[2.5, 0, -4.5]} scale={[1.1, 1.1, 1.1]}>
      {/* Dynamic abstract torus knot wireframe */}
      <torusKnotGeometry args={[3, 0.7, 80, 12]} />
      <meshBasicMaterial 
        color="#2FD8D5" 
        wireframe 
        transparent 
        opacity={0.02} // Super faint to prevent interfering with content text
      />
    </mesh>
  );
}

export default function BackgroundScene() {
  return (
    <div 
      style={{ 
        position: "fixed", 
        top: 0, 
        left: 0, 
        width: "100vw", 
        height: "100vh", 
        zIndex: -1, // Keep behind all layout cards
        pointerEvents: "none", // Click-through enabled
        opacity: 0.85
      }}
      aria-hidden="true"
    >
      <Canvas 
        camera={{ position: [0, 0, 6], fov: 45 }}
        shadows={false}
        gl={{ antialias: true, alpha: true, powerPreference: "low-power" }} // Optimize for battery/low-power
      >
        <ambientLight intensity={0.5} />
        <FloatingNetwork />
        <HolographicCore />
      </Canvas>
    </div>
  );
}
