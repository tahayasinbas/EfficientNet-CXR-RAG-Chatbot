import React, { useState } from 'react';
import PropTypes from 'prop-types';
import { FiCopy, FiCheck } from 'react-icons/fi';
import { FaRobot, FaPills } from 'react-icons/fa';
import { AiOutlineFileText } from 'react-icons/ai';

const RAGAssistant = ({ ragData }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = async (text) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Kopyalama başarısız:', err);
    }
  };

  if (!ragData) {
    return (
      <div className="bg-slate-darker rounded-lg p-6 text-center">
        <FaRobot className="text-gray-600 text-5xl mx-auto mb-3" />
        <p className="text-gray-500 text-sm">
          AI analiz sonucu bekleniyor...
        </p>
      </div>
    );
  }

  return (
    <div className="bg-slate-darker rounded-lg p-5 space-y-4 border border-gray-700">
      {/* Başlık */}
      <div className="flex items-center justify-between border-b border-gray-700 pb-3">
        <div className="flex items-center gap-2">
          <FaRobot className="text-neon-teal text-xl" />
          <h3 className="text-lg font-bold text-white">RAG Asistanı</h3>
        </div>
        <button
          onClick={() => handleCopy(`${ragData.analysis}\n\n${ragData.treatment}`)}
          className={`transition-colors p-2 ${copied ? 'text-safe-green' : 'text-gray-400 hover:text-white'}`}
          title={copied ? 'Kopyalandı!' : 'Tümünü Kopyala'}
          aria-label="Metni kopyala"
        >
          {copied ? <FiCheck size={18} /> : <FiCopy size={18} />}
        </button>
      </div>

      {/* AI Analizi */}
      {ragData.analysis && (
        <div>
          <div className="flex items-center gap-2 mb-2">
            <AiOutlineFileText className="text-medical-blue" />
            <h4 className="text-sm font-semibold text-gray-300">
              AI Analizi
            </h4>
          </div>
          <div className="bg-slate-dark rounded-lg p-4 border border-gray-700">
            <p className="text-sm text-gray-300 leading-relaxed">
              {ragData.analysis}
            </p>
          </div>
        </div>
      )}

      {/* Tedavi Önerisi */}
      {ragData.treatment && (
        <div>
          <div className="flex items-center gap-2 mb-2">
            <FaPills className="text-safe-green" />
            <h4 className="text-sm font-semibold text-gray-300">
              Tedavi Protokolü
            </h4>
          </div>
          <div className="bg-slate-dark rounded-lg p-4 border border-gray-700">
            <p className="text-sm text-gray-300 leading-relaxed">
              {ragData.treatment}
            </p>
          </div>
        </div>
      )}

      {/* Disclaimer */}
      <div className="bg-yellow-900/20 border border-yellow-700/50 rounded-lg p-3 mt-4">
        <p className="text-xs text-yellow-200">
          ⚠️ Bu sonuçlar AI tarafından üretilmiştir. Kesin tanı için bir
          radyoloji uzmanına danışın.
        </p>
      </div>
    </div>
  );
};

RAGAssistant.propTypes = {
  ragData: PropTypes.shape({
    analysis: PropTypes.string,
    treatment: PropTypes.string,
  }),
};

RAGAssistant.defaultProps = {
  ragData: null,
};

export default RAGAssistant;
