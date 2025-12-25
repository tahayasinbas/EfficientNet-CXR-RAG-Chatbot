import React from 'react';
import { FiCpu, FiLoader } from 'react-icons/fi';
import { Button } from '../ui/button';
import { cn } from '../../lib/utils';

const AnalyzeButton = ({ onClick, loading, disabled }) => {
  return (
    <Button
      onClick={onClick}
      disabled={disabled || loading}
      size="lg"
      className={cn(
        "w-full h-14 text-base font-bold transition-all duration-300",
        "shadow-lg hover:shadow-xl hover:scale-[1.02]",
        "disabled:scale-100 disabled:shadow-md",
        "bg-gradient-to-br from-medical-blue-500 to-medical-blue-700 text-white",
        "hover:from-medical-blue-600 hover:to-medical-blue-800",
        "dark:shadow-medical-blue/20"
      )}
    >
      {loading ? (
        <>
          <FiLoader className="animate-spin" size={22} />
          <span>Analiz Ediliyor...</span>
        </>
      ) : (
        <>
          <FiCpu size={22} />
          <span>Görüntüyü Analiz Et</span>
        </>
      )}
    </Button>
  );
};

export default AnalyzeButton;
