import { Canvas } from "@react-three/fiber";
import { Float, OrbitControls, Environment } from "@react-three/drei";
import { memo } from "react";

// Memoize components to prevent unnecessary re-renders in React Fiber loop
const GlassDiagnosticCore = memo(function GlassDiagnosticCore() {
  return (
    <Float rotationIntensity={0.6} floatIntensity={1.4} speed={1.1}>
      {/* 3D Glassmorphic Medical Core Core Mesh */}
      <mesh position={[0, 0, 0]} castShadow receiveShadow>
        <boxGeometry args={[3.2, 2.0, 0.12]} />
        <meshPhysicalMaterial 
          color="#06b6d4" 
          metalness={0.9} 
          roughness={0.08}
          transmission={0.8}
          thickness={0.25}
          transparent
          opacity={0.3}
          clearcoat={1.0}
          clearcoatRoughness={0.1}
        />
      </mesh>
      
      {/* Outer Holographic Wireframe Core Layer */}
      <mesh position={[0, 0, 0]}>
        <boxGeometry args={[3.24, 2.04, 0.14]} />
        <meshBasicMaterial 
          color="#2FD8D5" 
          wireframe 
          transparent 
          opacity={0.15} 
        />
      </mesh>
    </Float>
  );
});

interface FloatingSphereProps {
  position: [number, number, number];
  glowColor: string;
  speed: number;
}

const FloatingSphere = memo(function FloatingSphere({ position, glowColor, speed }: FloatingSphereProps) {
  return (
    <Float rotationIntensity={0.8} floatIntensity={2.0} speed={speed}>
      <mesh position={position}>
        <sphereGeometry args={[0.22, 32, 32]} />
        <meshStandardMaterial 
          color={glowColor} 
          emissive={glowColor} 
          emissiveIntensity={0.8} 
          roughness={0.1} 
          metalness={0.5}
        />
      </mesh>
    </Float>
  );
});

export default function HeroScene() {
  return (
    <div style={{ width: "100%", height: "100%", minHeight: 480, position: "relative" }}>
      <Canvas 
        camera={{ position: [0, 0, 5.0], fov: 38 }} // Bring camera close to center the 3D core
        shadows={false}
        gl={{ antialias: true, alpha: true, powerPreference: "high-performance" }}
      >
        <ambientLight intensity={0.8} />
        <directionalLight position={[3, 5, 2]} intensity={1.5} />
        <directionalLight position={[-3, -5, -2]} intensity={0.5} color="#3b82f6" />
        <Environment preset="city" />
        
        {/* Render the floating glass core and ambient spheres */}
        <GlassDiagnosticCore />
        <FloatingSphere position={[-2.0, 1.2, 0.5]} glowColor="#2FD8D5" speed={1.2} />
        <FloatingSphere position={[2.0, -1.2, -0.5]} glowColor="#fbbf24" speed={1.0} />
        
        <OrbitControls enableZoom={false} autoRotate autoRotateSpeed={0.5} enableDamping />
      </Canvas>
    </div>
  );
}
