import React from 'react';
import PropTypes from 'prop-types';
import { FiX, FiActivity, FiAlertTriangle, FiInfo, FiCpu, FiCheckCircle } from 'react-icons/fi';
import { cn } from '../../lib/utils';

const getRiskColor = (confidence) => {
  if (confidence >= 0.7) return 'bg-status-error';
  if (confidence >= 0.3) return 'bg-status-warning';
  return 'bg-status-success';
};

const getRiskTextColor = (confidence) => {
  if (confidence >= 0.7) return 'text-status-error';
  if (confidence >= 0.3) return 'text-status-warning';
  return 'text-status-success';
};

const getRiskLevel = (confidence) => {
  if (confidence >= 0.7) return 'Yüksek Risk';
  if (confidence >= 0.3) return 'Orta Risk';
  return 'Düşük Risk';
};

const getRiskDescription = (confidence) => {
  if (confidence >= 0.7) {
    return 'Model bu hastalığın varlığına yüksek güvenle işaret ediyor. İleri tetkik ve uzman konsültasyonu önerilir.';
  }
  if (confidence >= 0.3) {
    return 'Model orta düzeyde bir olasılık saptadı. Klinik korelasyon ve ek görüntüleme değerlendirilmelidir.';
  }
  return 'Model düşük bir olasılık saptadı. Ancak klinik bulguların da göz önünde bulundurulması önemlidir.';
};

const DiagnosisDetailModal = ({ diagnosis, allResults, patientInfo, onClose, showAllResults = false }) => {
  if (!diagnosis) return null;

  const percentage = (diagnosis.confidence * 100).toFixed(1);
  const riskLevel = getRiskLevel(diagnosis.confidence);
  const riskDescription = getRiskDescription(diagnosis.confidence);

  // En yüksek 5 hastalığı al veya showAllResults true ise hepsini göster
  const top5Diseases = allResults ? allResults.slice(0, 5) : [];
  const displayDiseases = showAllResults ? allResults : top5Diseases;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm animate-fade-in">
      <div className="bg-surface rounded-xl shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-hidden animate-scale-in">
        {/* Header */}
        <div className="p-6 border-b border-border-light">
          <div className="flex items-start justify-between">
            <div className="flex-1">
              <div className="flex items-center gap-3 mb-2">
                <div className="w-10 h-10 rounded-lg bg-medical-blue/10 flex items-center justify-center">
                  <FiActivity className="text-medical-blue" size={20} />
                </div>
                <h2 className="text-2xl font-bold text-text-primary">
                  {diagnosis.disease_name}
                </h2>
              </div>
              <p className="text-sm text-text-secondary">
                Detaylı Analiz Raporu
              </p>
            </div>
            <button
              onClick={onClose}
              className="text-text-tertiary hover:text-text-primary transition-colors p-2 hover:bg-surface-elevated rounded-lg"
            >
              <FiX size={24} />
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto max-h-[calc(90vh-200px)] space-y-6">
          {/* Confidence Score */}
          <div className="bg-surface-elevated rounded-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-semibold text-text-secondary uppercase tracking-wider">
                Güven Skoru
              </h3>
              <span className={cn("text-4xl font-bold", getRiskTextColor(diagnosis.confidence))}>
                {percentage}%
              </span>
            </div>
            <div className="w-full bg-bg-tertiary rounded-full h-3 overflow-hidden mb-3">
              <div
                className={cn(
                  "h-3 rounded-full transition-all duration-500",
                  getRiskColor(diagnosis.confidence)
                )}
                style={{ width: `${percentage}%` }}
              ></div>
            </div>
            <div className="flex items-center gap-2">
              <FiAlertTriangle className={getRiskTextColor(diagnosis.confidence)} size={18} />
              <span className={cn("text-sm font-semibold", getRiskTextColor(diagnosis.confidence))}>
                {riskLevel}
              </span>
            </div>
          </div>

          {/* Risk Description */}
          <div className="bg-surface-elevated rounded-lg p-5">
            <div className="flex items-start gap-3">
              <FiInfo className="text-medical-blue mt-0.5 flex-shrink-0" size={18} />
              <div>
                <h3 className="text-sm font-semibold text-text-primary mb-2">
                  Klinik Değerlendirme
                </h3>
                <p className="text-sm text-text-secondary leading-relaxed">
                  {riskDescription}
                </p>
              </div>
            </div>
          </div>

          {/* En Yüksek 5 Hastalık Karşılaştırması */}
          {displayDiseases.length > 0 && (
            <div className="bg-surface-elevated rounded-lg p-5">
              <div className="flex items-start gap-3 mb-4">
                <FiActivity className="text-medical-blue mt-0.5 flex-shrink-0" size={18} />
                <div className="flex-1">
                  <h3 className="text-sm font-semibold text-text-primary mb-3">
                    {showAllResults ? `Tüm Hastalık Olasılıkları (${displayDiseases.length})` : 'En Yüksek 5 Hastalık Olasılığı'}
                  </h3>
                  <div className="space-y-3 max-h-96 overflow-y-auto pr-2">
                    {displayDiseases.map((disease, index) => {
                      const diseasePercentage = (disease.confidence * 100).toFixed(1);
                      const isCurrentDiagnosis = disease.disease_name === diagnosis.disease_name;

                      return (
                        <div key={disease.disease_name} className={cn(
                          "p-3 rounded-lg transition-all",
                          isCurrentDiagnosis ? "bg-medical-blue/10 border border-medical-blue/30" : "bg-bg-secondary"
                        )}>
                          <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center gap-2">
                              <span className={cn(
                                "text-xs font-bold w-6 h-6 rounded-full flex items-center justify-center",
                                isCurrentDiagnosis ? "bg-medical-blue text-white" : "bg-bg-tertiary text-text-tertiary"
                              )}>
                                {index + 1}
                              </span>
                              <span className={cn(
                                "text-sm font-medium",
                                isCurrentDiagnosis ? "text-medical-blue" : "text-text-primary"
                              )}>
                                {disease.disease_name}
                              </span>
                            </div>
                            <span className={cn(
                              "text-base font-bold",
                              getRiskTextColor(disease.confidence)
                            )}>
                              {diseasePercentage}%
                            </span>
                          </div>
                          <div className="w-full bg-bg-tertiary rounded-full h-2 overflow-hidden">
                            <div
                              className={cn(
                                "h-2 rounded-full transition-all duration-500",
                                getRiskColor(disease.confidence)
                              )}
                              style={{ width: `${diseasePercentage}%` }}
                            ></div>
                          </div>
                          {isCurrentDiagnosis && (
                            <div className="mt-2 text-xs text-medical-blue font-medium">
                              ● Seçili Hastalık
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Model Information */}
          <div className="bg-surface-elevated rounded-lg p-5">
            <div className="flex items-start gap-3 mb-4">
              <FiCpu className="text-medical-blue mt-0.5 flex-shrink-0" size={18} />
              <div className="flex-1">
                <h3 className="text-sm font-semibold text-text-primary mb-3">
                  Model Detayları
                </h3>
                <div className="space-y-2.5">
                  <InfoRow label="Model Mimarisi" value="EfficientNet-B3 + Multimodal" />
                  <InfoRow label="Eğitim Dataset" value="ChestX-ray14 (NIH)" />
                  <InfoRow label="Model Performansı" value="AUC: 0.89 (Validation)" />
                </div>
              </div>
            </div>
          </div>

          {/* Patient & Analysis Info */}
          {patientInfo && (
            <div className="bg-surface-elevated rounded-lg p-5">
              <div className="flex items-start gap-3 mb-4">
                <FiCheckCircle className="text-status-success mt-0.5 flex-shrink-0" size={18} />
                <div className="flex-1">
                  <h3 className="text-sm font-semibold text-text-primary mb-3">
                    Analiz Özellikleri
                  </h3>
                  <div className="space-y-2.5">
                    <InfoRow label="Yaş" value={`${patientInfo.age} yaş`} />
                    <InfoRow label="Cinsiyet" value={patientInfo.gender === 'M' ? 'Erkek' : 'Kadın'} />
                    <InfoRow label="Görüntü Pozisyonu" value={patientInfo.position === 'PA' ? 'Posteroanterior' : 'Anteroposterior'} />
                    <InfoRow label="Görselleştirme" value="GradCAM Attention Maps (Yakında)" />
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Medical Disclaimer */}
          <div className="bg-status-warning/10 border border-status-warning/30 rounded-lg p-4">
            <div className="flex gap-3">
              <FiAlertTriangle className="text-status-warning mt-0.5 flex-shrink-0" size={18} />
              <div>
                <h4 className="text-sm font-semibold text-status-warning mb-1">
                  Önemli Uyarı
                </h4>
                <p className="text-xs text-text-secondary leading-relaxed">
                  Bu analiz yardımcı bir tanı aracıdır ve kesin teşhis yerine geçmez.
                  Tüm bulgular deneyimli bir radyolog veya klinisyen tarafından değerlendirilmelidir.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-border-light bg-surface-elevated">
          <button
            onClick={onClose}
            className="w-full px-4 py-2.5 bg-medical-blue hover:bg-medical-blue-dark text-white font-medium rounded-lg transition-colors"
          >
            Kapat
          </button>
        </div>
      </div>
    </div>
  );
};

const InfoRow = ({ label, value }) => (
  <div className="flex justify-between items-center py-1">
    <span className="text-xs text-text-tertiary">{label}:</span>
    <span className="text-sm text-text-primary font-medium">{value}</span>
  </div>
);

InfoRow.propTypes = {
  label: PropTypes.string.isRequired,
  value: PropTypes.string.isRequired,
};

DiagnosisDetailModal.propTypes = {
  diagnosis: PropTypes.shape({
    disease_name: PropTypes.string.isRequired,
    confidence: PropTypes.number.isRequired,
  }),
  allResults: PropTypes.arrayOf(
    PropTypes.shape({
      disease_name: PropTypes.string.isRequired,
      confidence: PropTypes.number.isRequired,
    })
  ),
  patientInfo: PropTypes.shape({
    age: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
    gender: PropTypes.string,
    position: PropTypes.string,
  }),
  onClose: PropTypes.func.isRequired,
  showAllResults: PropTypes.bool,
};

export default DiagnosisDetailModal;
