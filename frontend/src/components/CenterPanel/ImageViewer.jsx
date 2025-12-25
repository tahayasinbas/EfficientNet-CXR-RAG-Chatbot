import React, { useState } from 'react';
import PropTypes from 'prop-types';
import FloatingToolbar from './FloatingToolbar';
import HeatmapSlider from './HeatmapSlider';
import { FiImage } from 'react-icons/fi';
import { ZOOM } from '../../constants';
import { cn } from '../../lib/utils';

const ImageViewer = ({ image, heatmapImage }) => {
  const [zoom, setZoom] = useState(ZOOM.DEFAULT);
  const [inverted, setInverted] = useState(false);
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [heatmapOpacity, setHeatmapOpacity] = useState(50);

  const handleZoomIn = () => setZoom((z) => Math.min(z + ZOOM.STEP, ZOOM.MAX));
  const handleZoomOut = () => setZoom((z) => Math.max(z - ZOOM.STEP, ZOOM.MIN));
  const handleInvert = () => setInverted((i) => !i);
  const handleToggleHeatmap = () => setShowHeatmap((s) => !s);
  const handleReset = () => {
    setZoom(ZOOM.DEFAULT);
    setInverted(false);
    setShowHeatmap(false);
    setHeatmapOpacity(50);
  };

  return (
    <div className="relative h-full flex items-center justify-center bg-white/70 backdrop-blur-xl rounded-xl border border-white/40 shadow-lg shadow-blue-500/10 overflow-hidden animate-fade-in">
      {!image ? (
        <div className="text-center text-gray-500 animate-fade-in">
          <FiImage className="mx-auto text-5xl mb-4 text-blue-500/40" />
          <p className="font-semibold text-base text-gray-700">Görüntü Bekleniyor</p>
          <p className="text-sm text-gray-500">Analiz için sol panelden bir görüntü yükleyin.</p>
        </div>
      ) : (
        <>
          <div
            className="absolute inset-0 flex items-center justify-center transition-transform duration-300 ease-out"
            style={{ transform: `scale(${zoom})` }}
          >
            <img
              src={image}
              alt="Röntgen görüntüsü"
              className={cn(
                "max-w-full max-h-full object-contain rounded-lg shadow-2xl",
                "transition-all duration-300",
                inverted ? 'invert' : 'invert-0'
              )}
            />
            {showHeatmap && heatmapImage && (
              <img
                src={heatmapImage}
                alt="AI heatmap"
                className="absolute top-0 left-0 w-full h-full object-contain pointer-events-none transition-opacity duration-300 rounded-lg"
                style={{ opacity: heatmapOpacity / 100, mixBlendMode: 'color' }}
              />
            )}
          </div>

          <FloatingToolbar
            onZoomIn={handleZoomIn}
            onZoomOut={handleZoomOut}
            onReset={handleReset}
            onInvert={handleInvert}
            isInverted={inverted}
            showHeatmap={showHeatmap}
            onToggleHeatmap={handleToggleHeatmap}
            canShowHeatmap={!!heatmapImage}
          />

          {heatmapImage && (
            <HeatmapSlider
              opacity={heatmapOpacity}
              onChange={setHeatmapOpacity}
              visible={showHeatmap}
            />
          )}
        </>
      )}
    </div>
  );
};

ImageViewer.propTypes = {
  image: PropTypes.string,
  heatmapImage: PropTypes.string,
};

export default ImageViewer;

