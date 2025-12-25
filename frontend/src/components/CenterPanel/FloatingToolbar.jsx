import React from 'react';
import PropTypes from 'prop-types';
import { FiZoomIn, FiZoomOut, FiRefreshCw, FiSun, FiMoon, FiEye } from 'react-icons/fi';
import { Button } from '../ui/button';
import { cn } from '../../lib/utils';

const ToolbarButton = ({ onClick, title, children, isActive }) => (
  <Button
    onClick={onClick}
    variant="ghost"
    size="icon"
    className={cn(
      "w-10 h-10 rounded-full text-text-secondary hover:text-text-primary hover:bg-black/10 dark:hover:bg-white/10",
      isActive && "bg-medical-blue/10 text-medical-blue hover:text-medical-blue"
    )}
    title={title}
    aria-label={title}
  >
    {children}
  </Button>
);

const FloatingToolbar = ({
  onZoomIn,
  onZoomOut,
  onReset,
  onInvert,
  isInverted,
  showHeatmap,
  onToggleHeatmap,
  canShowHeatmap,
}) => {
  return (
    <div className="absolute bottom-5 left-1/2 -translate-x-1/2 z-20">
      <div className="flex items-center gap-2 rounded-full border border-white/10 bg-gray-500/10 p-1.5 backdrop-blur-sm shadow-lg">
        <ToolbarButton onClick={onZoomIn} title="Yakınlaştır">
          <FiZoomIn size={18} />
        </ToolbarButton>
        <ToolbarButton onClick={onZoomOut} title="Uzaklaştır">
          <FiZoomOut size={18} />
        </ToolbarButton>
        <div className="w-px h-5 bg-white/10 mx-1"></div>
        <ToolbarButton onClick={onReset} title="Sıfırla">
          <FiRefreshCw size={18} />
        </ToolbarButton>
        <ToolbarButton onClick={onInvert} title="Renkleri Ters Çevir" isActive={isInverted}>
          {isInverted ? <FiSun size={18} /> : <FiMoon size={18} />}
        </ToolbarButton>
        {canShowHeatmap && (
          <>
            <div className="w-px h-5 bg-white/10 mx-1"></div>
            <ToolbarButton
              onClick={onToggleHeatmap}
              title="AI Isı Haritasını Göster"
              isActive={showHeatmap}
            >
              <FiEye size={18} />
            </ToolbarButton>
          </>
        )}
      </div>
    </div>
  );
};

FloatingToolbar.propTypes = {
  onZoomIn: PropTypes.func.isRequired,
  onZoomOut: PropTypes.func.isRequired,
  onReset: PropTypes.func.isRequired,
  onInvert: PropTypes.func.isRequired,
  isInverted: PropTypes.bool.isRequired,
  showHeatmap: PropTypes.bool.isRequired,
  onToggleHeatmap: PropTypes.func.isRequired,
  canShowHeatmap: PropTypes.bool,
};

FloatingToolbar.defaultProps = {
  canShowHeatmap: false,
};

export default FloatingToolbar;
