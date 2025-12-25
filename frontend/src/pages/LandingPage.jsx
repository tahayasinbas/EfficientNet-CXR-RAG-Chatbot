import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

const LandingPage = () => {
  const navigate = useNavigate();
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const [particles, setParticles] = useState([]);

  useEffect(() => {
    // Generate random floating particles
    const generateParticles = () => {
      const newParticles = [];
      for (let i = 0; i < 20; i++) {
        newParticles.push({
          id: i,
          x: Math.random() * 100,
          y: Math.random() * 100,
          size: Math.random() * 4 + 2,
          duration: Math.random() * 10 + 10,
          delay: Math.random() * 5,
        });
      }
      setParticles(newParticles);
    };
    generateParticles();
  }, []);

  useEffect(() => {
    const handleMouseMove = (e) => {
      setMousePosition({
        x: (e.clientX / window.innerWidth) * 100,
        y: (e.clientY / window.innerHeight) * 100,
      });
    };
    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  const handleExplore = () => {
    navigate('/dashboard');
  };

  return (
    <div className="relative w-full h-screen overflow-hidden bg-gradient-to-br from-pink-100 via-blue-50 to-cyan-100">
      <style>{`
        @keyframes float {
          0%, 100% { transform: translateY(0px) translateX(0px); }
          25% { transform: translateY(-20px) translateX(10px); }
          50% { transform: translateY(-10px) translateX(-10px); }
          75% { transform: translateY(-30px) translateX(5px); }
        }

        @keyframes pulse-glow {
          0%, 100% { opacity: 0.3; transform: scale(1); }
          50% { opacity: 0.6; transform: scale(1.1); }
        }

        @keyframes gradient-shift {
          0% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
          100% { background-position: 0% 50%; }
        }

        @keyframes wave {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }

        @keyframes ripple {
          0% { transform: scale(0.8); opacity: 0.8; }
          100% { transform: scale(1.2); opacity: 0; }
        }

        @keyframes scan-line {
          0% { top: 0%; opacity: 0; }
          5% { opacity: 0.8; }
          50% { top: 100%; opacity: 0.8; }
          55% { opacity: 0; }
          100% { top: 100%; opacity: 0; }
        }

        .animated-gradient {
          background: linear-gradient(-45deg, #ffc0cb, #87ceeb, #00ffff, #e0b0ff);
          background-size: 400% 400%;
          animation: gradient-shift 15s ease infinite;
        }

        .lung-image-container {
          filter: brightness(1.05) contrast(1.1);
        }

        .lung-image-container:hover {
          filter: brightness(1.15) contrast(1.15);
        }
      `}</style>

      {/* Animated Background Gradients */}
      <div className="absolute inset-0 animated-gradient opacity-50"></div>

      {/* Floating Particles */}
      {particles.map((particle) => (
        <div
          key={particle.id}
          className="absolute rounded-full bg-white/40"
          style={{
            left: `${particle.x}%`,
            top: `${particle.y}%`,
            width: `${particle.size}px`,
            height: `${particle.size}px`,
            animation: `float ${particle.duration}s ease-in-out infinite`,
            animationDelay: `${particle.delay}s`,
          }}
        />
      ))}

      {/* Dynamic Gradient Layers that follow mouse */}
      <div
        className="absolute top-0 left-0 w-full h-full pointer-events-none"
        style={{
          background: `radial-gradient(circle at ${mousePosition.x}% ${mousePosition.y}%, rgba(255, 192, 203, 0.3) 0%, transparent 50%)`,
          transition: 'background 0.3s ease',
        }}
      />

      {/* Content Container */}
      <div className="relative z-10 flex flex-col items-center justify-center h-full text-center px-6">
        {/* Top Badge */}
        <div className="mb-8 transform hover:scale-110 transition-transform duration-300">
          <span className="inline-flex items-center gap-2 px-6 py-3 bg-white/80 backdrop-blur-sm rounded-full text-gray-700 text-sm font-medium shadow-lg hover:shadow-xl transition-shadow">
            <span className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></span>
            AI-Powered Medical Analysis
          </span>
        </div>

        {/* Main Heading with gradient text */}
        <h1 className="text-5xl md:text-7xl font-thin text-gray-800 mb-6 leading-tight relative z-20 hover:scale-105 transition-transform duration-500">
          ChestAI — <span className="bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 bg-clip-text text-transparent">Intelligent Biomarker</span>
          <br />
          Monitoring Platform
        </h1>

        {/* Real Lung Image Container */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[900px] h-[700px] z-0">
          {/* Multi-layered Background Glow Effects */}
          <div className="absolute inset-0">
            <div
              className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[700px] h-[600px]"
              style={{
                background: 'radial-gradient(ellipse, rgba(59, 130, 246, 0.2) 20%, rgba(168, 85, 247, 0.15) 50%, transparent 80%)',
                animation: 'pulse-glow 5s ease-in-out infinite',
                filter: 'blur(50px)',
              }}
            />
            <div
              className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[500px]"
              style={{
                background: 'radial-gradient(ellipse, rgba(236, 72, 153, 0.15) 30%, transparent 70%)',
                animation: 'pulse-glow 7s ease-in-out infinite 1s',
                filter: 'blur(60px)',
              }}
            />
          </div>

          {/* Real Lung Image with Interactive-like Effects */}
          <div
            className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 lung-image-container group cursor-pointer"
            style={{
              width: '1000px',
              height: '750px',
              transform: `translate(-50%, -50%) translate(${mousePosition.x * 0.02}px, ${mousePosition.y * 0.02}px)`,
              transition: 'transform 0.3s ease',
            }}
          >
            <img
              src="/gogus_resmi.png"
              alt="Chest and Lung Anatomy"
              className="w-full h-full transition-all duration-500 group-hover:scale-105"
              style={{
                objectFit: 'cover',
                objectPosition: 'center',
                clipPath: 'inset(8% 12% 8% 12%)',
                filter: 'drop-shadow(0 0 40px rgba(59, 130, 246, 0.6)) drop-shadow(0 0 60px rgba(168, 85, 247, 0.4))',
                animation: 'pulse-glow 4s ease-in-out infinite',
              }}
            />

            {/* Interactive Overlay Gradients */}
            <div
              className="absolute inset-0 pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity duration-500"
              style={{
                background: `radial-gradient(circle at ${mousePosition.x}% ${mousePosition.y}%, rgba(59, 130, 246, 0.2) 0%, rgba(168, 85, 247, 0.1) 40%, transparent 70%)`,
              }}
            />

            {/* Glowing Border Effect on Hover */}
            <div
              className="absolute inset-0 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-500"
              style={{
                boxShadow: '0 0 80px rgba(59, 130, 246, 0.5), inset 0 0 80px rgba(168, 85, 247, 0.3)',
                animation: 'pulse-glow 3s ease-in-out infinite',
              }}
            />

            {/* Animated Scanning Line Effect */}
            <div
              className="absolute left-0 right-0 h-[2px] bg-gradient-to-r from-transparent via-blue-400 to-transparent opacity-60"
              style={{
                animation: 'scan-line 4s ease-in-out infinite',
                top: '0%',
              }}
            />
          </div>

          {/* Subtle Wave Lines - Simplified */}
          <div className="absolute inset-0 opacity-30">
            {[0, 1].map((i) => (
              <div
                key={i}
                className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 rounded-full border border-blue-300/15"
                style={{
                  width: `${650 + i * 150}px`,
                  height: `${550 + i * 120}px`,
                  animation: `wave ${25 + i * 8}s linear infinite ${i * 3}s`,
                }}
              />
            ))}
          </div>

          {/* Enhanced Glow effects for lungs */}
          <div className="absolute top-0 left-0 w-full h-full pointer-events-none">
            {/* Left lung glow */}
            <div
              className="absolute left-[15%] top-1/2 -translate-y-1/2 w-[350px] h-[500px] bg-blue-500/15 rounded-full blur-3xl"
              style={{ animation: 'pulse-glow 6s ease-in-out infinite' }}
            />
            {/* Right lung glow */}
            <div
              className="absolute right-[15%] top-1/2 -translate-y-1/2 w-[350px] h-[500px] bg-purple-500/15 rounded-full blur-3xl"
              style={{ animation: 'pulse-glow 6s ease-in-out infinite 1.5s' }}
            />
            {/* Center glow */}
            <div
              className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[600px] bg-cyan-400/8 rounded-full blur-3xl"
              style={{ animation: 'pulse-glow 7s ease-in-out infinite 3s' }}
            />
          </div>
        </div>

        {/* Subtitle */}
        <div className="relative z-20">
          <p className="text-xl md:text-2xl text-gray-600 mb-4 max-w-3xl font-light hover:text-gray-800 transition-colors duration-300">
            Tracking, analyzing, and visualizing patient biomarkers in real time.
          </p>
          <p className="text-lg md:text-xl text-gray-600 mb-12 font-light hover:text-gray-800 transition-colors duration-300">
            Empowering smarter, data-driven healthcare.
          </p>
        </div>

        {/* Tags with hover effects */}
        <div className="flex flex-wrap gap-4 justify-center mb-12 text-gray-500 text-sm relative z-20">
          {['product design', 'branding', 'web design', 'mobile design'].map((tag, index) => (
            <span
              key={tag}
              className="px-4 py-2 bg-white/50 backdrop-blur-sm rounded-full hover:bg-white/70 hover:scale-110 transition-all duration-300 cursor-pointer hover:shadow-lg"
              style={{ animationDelay: `${index * 0.1}s` }}
            >
              {tag}
            </span>
          ))}
        </div>

        {/* CTA Button with enhanced effects */}
        <button
          onClick={handleExplore}
          className="group relative z-20 px-10 py-4 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white text-lg font-medium rounded-full shadow-xl transition-all duration-300 transform hover:scale-110 hover:shadow-2xl overflow-hidden"
        >
          {/* Button ripple effect */}
          <span className="absolute inset-0 bg-white opacity-0 group-hover:opacity-20 rounded-full transition-opacity duration-300"></span>

          <span className="relative flex items-center gap-2">
            Uygulamayı Keşfet
            <svg
              className="w-5 h-5 transform group-hover:translate-y-1 transition-transform duration-300"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
            </svg>
          </span>
        </button>
      </div>

    </div>
  );
};

export default LandingPage;
