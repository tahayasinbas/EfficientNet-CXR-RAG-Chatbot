import React from 'react';
import Header from './Header';

const MainLayout = ({ children }) => {
  return (
    <div className="h-screen flex flex-col overflow-hidden relative">
      {/* Animated Gradient Background */}
      <div className="fixed inset-0 bg-gradient-to-br from-pink-50 via-blue-50 to-cyan-50 -z-20"></div>

      {/* Animated Gradient Overlay */}
      <div
        className="fixed inset-0 -z-10 opacity-60"
        style={{
          background: 'linear-gradient(-45deg, rgba(255, 192, 203, 0.3), rgba(135, 206, 235, 0.3), rgba(0, 255, 255, 0.3), rgba(224, 176, 255, 0.3))',
          backgroundSize: '400% 400%',
          animation: 'gradient-shift 15s ease infinite',
        }}
      ></div>

      {/* Floating Particles */}
      <div className="fixed inset-0 -z-10 pointer-events-none">
        {[...Array(15)].map((_, i) => (
          <div
            key={i}
            className="absolute rounded-full bg-white/30"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              width: `${Math.random() * 4 + 2}px`,
              height: `${Math.random() * 4 + 2}px`,
              animation: `float ${Math.random() * 10 + 10}s ease-in-out infinite`,
              animationDelay: `${Math.random() * 5}s`,
            }}
          />
        ))}
      </div>

      <Header />
      <main className="flex-1 container mx-auto px-4 py-6 max-w-[1920px] overflow-hidden">
        {children}
      </main>

      <style>{`
        @keyframes gradient-shift {
          0% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
          100% { background-position: 0% 50%; }
        }

        @keyframes float {
          0%, 100% { transform: translateY(0px) translateX(0px); }
          25% { transform: translateY(-20px) translateX(10px); }
          50% { transform: translateY(-10px) translateX(-10px); }
          75% { transform: translateY(-30px) translateX(5px); }
        }
      `}</style>
    </div>
  );
};

export default MainLayout;
