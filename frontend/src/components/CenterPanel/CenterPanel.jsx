import React from 'react';
import ImageViewer from './ImageViewer';

const CenterPanel = ({ image, heatmapImage }) => {
  return (
    <div className="h-full">
      <ImageViewer image={image} heatmapImage={heatmapImage} />
    </div>
  );
};

export default CenterPanel;
