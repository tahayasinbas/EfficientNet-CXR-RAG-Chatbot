import React from 'react';
import { FiCpu } from 'react-icons/fi';
import { GiHealthNormal } from 'react-icons/gi';

const Header = () => {
  return (
    <header className="bg-white/80 backdrop-blur-lg border-b border-purple-100/50 px-8 py-4 sticky top-0 z-50 shadow-sm">
      <div className="flex items-center justify-between w-full max-w-none">
        {/* Logo and Project Name */}
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 bg-gradient-to-br from-blue-500 via-purple-500 to-pink-500 rounded-xl flex items-center justify-center shadow-lg transform hover:scale-105 transition-transform duration-300">
            <GiHealthNormal className="text-white text-3xl" />
          </div>
          <div>
            <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 bg-clip-text text-transparent tracking-tight">
              ChestAI — Radiology Assistant
            </h1>
            <p className="text-xs text-gray-500 flex items-center gap-2 mt-0.5 font-medium">
              <FiCpu size={13} className="text-blue-500" />
              <span>Powered by EfficientNet-B4 & NIH Dataset</span>
            </p>
          </div>
        </div>

        {/* Status Indicator and User */}
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2.5 bg-green-50 px-4 py-2 rounded-full border border-green-200">
            <div className="relative flex items-center justify-center w-3 h-3">
              <div className="absolute inline-flex w-full h-full bg-green-500 rounded-full opacity-75 animate-ping"></div>
              <div className="relative inline-flex w-2 h-2 bg-green-500 rounded-full"></div>
            </div>
            <span className="text-sm text-green-700 font-semibold">System Online</span>
          </div>

          {/* User Avatar */}
          <div className="w-11 h-11 bg-gradient-to-br from-blue-500 via-purple-500 to-pink-500 rounded-full flex items-center justify-center text-white font-bold text-sm shadow-lg cursor-pointer ring-2 ring-offset-2 ring-offset-white/80 ring-purple-300 hover:ring-purple-400 hover:scale-105 transition-all duration-300">
            <span>TA</span>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
