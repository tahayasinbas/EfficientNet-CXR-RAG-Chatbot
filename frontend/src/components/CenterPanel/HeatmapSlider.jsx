import React from 'react';
import PropTypes from 'prop-types';
import { cn } from '../../lib/utils';
import { FiSun, FiThermometer } from 'react-icons/fi';

const HeatmapSlider = ({ opacity, onChange, visible }) => {
  return (
    <div
      className={cn(
        "absolute bottom-20 left-1/2 -translate-x-1/2 w-56 z-20 transition-all duration-300",
        visible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4 pointer-events-none"
      )}
    >
      <div className="rounded-lg border border-white/10 bg-gray-500/10 p-2.5 backdrop-blur-sm shadow-lg">
        <div className="flex items-center gap-2">
          <FiSun size={14} className="text-text-tertiary" />
          <input
            type="range"
            min="0"
            max="100"
            value={opacity}
            onChange={(e) => onChange(parseInt(e.target.value, 10))}
            className="w-full h-1.5 rounded-full appearance-none cursor-pointer slider-thumb"
            style={{
              background: `linear-gradient(to right, hsl(var(--primary)) 0%, hsl(var(--primary)) ${opacity}%, hsl(var(--border)) ${opacity}%, hsl(var(--border)) 100%)`,
            }}
          />
          <FiThermometer size={14} className="text-text-tertiary" />
        </div>
      </div>
    </div>
  );
};

HeatmapSlider.propTypes = {
  opacity: PropTypes.number.isRequired,
  onChange: PropTypes.func.isRequired,
  visible: PropTypes.bool.isRequired,
};

export default HeatmapSlider;

