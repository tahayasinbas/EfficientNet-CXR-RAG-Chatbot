import React from 'react';
import PropTypes from 'prop-types';
import { Card, CardContent } from '../ui/card';
import UploadZone from './UploadZone';
import PatientForm from './PatientForm';
import AnalyzeButton from './AnalyzeButton';
import { FiLayout, FiUser, FiUploadCloud, FiChevronLeft, FiChevronRight } from 'react-icons/fi';

const LeftPanel = ({
  selectedImage,
  onImageSelect,
  patientData,
  setPatientData,
  onAnalyze,
  loading,
  isCollapsed,
  onToggleCollapse,
}) => {
  if (isCollapsed) {
    return (
      <div className="h-full flex items-start justify-center pt-4">
        <button
          onClick={() => onToggleCollapse(false)}
          className="p-2 rounded-lg bg-white/70 backdrop-blur-xl hover:bg-white/90 border border-purple-200/50 text-blue-600 transition-all shadow-lg hover:shadow-blue-500/30"
          aria-label="Paneli Aç"
          title="Kontrol Panelini Aç"
        >
          <FiChevronRight size={20} />
        </button>
      </div>
    );
  }

  return (
    <Card className="h-full flex flex-col animate-fade-in">
      {/* Kompakt Header */}
      <div className="px-4 py-3 border-b border-gray-200/50">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center shadow-md">
              <FiLayout className="text-white" size={14} />
            </div>
            <h2 className="text-sm font-semibold text-gray-900">Kontrol Paneli</h2>
          </div>
          <button
            onClick={() => onToggleCollapse(true)}
            className="p-1.5 rounded-lg hover:bg-gray-100/50 text-gray-500 hover:text-gray-700 transition-colors"
            aria-label="Paneli Gizle"
            title="Paneli Gizle"
          >
            <FiChevronLeft size={16} />
          </button>
        </div>
      </div>

      {/* Main Content */}
      <CardContent className="p-4 flex-1 flex flex-col gap-4">
        {/* Upload Section */}
        <div className="space-y-2">
          <h3 className="text-xs font-semibold text-gray-600 flex items-center gap-1.5 uppercase tracking-wider">
            <FiUploadCloud size={12} className="text-blue-500" />
            <span>Görüntü Yükle</span>
          </h3>
          <UploadZone
            onImageSelect={onImageSelect}
            selectedImage={selectedImage}
          />
        </div>

        {/* Patient Info Section */}
        <div className="space-y-2">
          <h3 className="text-xs font-semibold text-gray-600 flex items-center gap-1.5 uppercase tracking-wider">
            <FiUser size={12} className="text-purple-600" />
            <span>Hasta Bilgileri</span>
          </h3>
          <PatientForm patientData={patientData} setPatientData={setPatientData} />
        </div>
      </CardContent>
      
      {/* Footer with Analyze Button */}
      <div className="p-4 border-t border-gray-200/50">
        <AnalyzeButton
          onClick={onAnalyze}
          loading={loading}
          disabled={!selectedImage}
        />
      </div>
    </Card>
  );
};

LeftPanel.propTypes = {
  selectedImage: PropTypes.object,
  onImageSelect: PropTypes.func.isRequired,
  patientData: PropTypes.object.isRequired,
  setPatientData: PropTypes.func.isRequired,
  onAnalyze: PropTypes.func.isRequired,
  loading: PropTypes.bool,
  isCollapsed: PropTypes.bool,
  onToggleCollapse: PropTypes.func.isRequired,
};

export default LeftPanel;
