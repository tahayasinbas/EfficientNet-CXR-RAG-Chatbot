import React, { useRef, useEffect } from 'react';
import PropTypes from 'prop-types';
import { Card, CardContent } from '../ui/card';
import StickyResultHeader from './StickyResultHeader';
import ChatMessage from './ChatMessage';
import ChatInput from './ChatInput';
import DiagnosisResults from './DiagnosisResults';
import { FiMessageSquare } from 'react-icons/fi';

const RightPanel = ({ results, loading, messages, onSendMessage, analysisComplete, patientInfo }) => {
  const messagesEndRef = useRef(null);
  const messagesContainerRef = useRef(null);
  const prevMessagesLength = useRef(messages.length);
  const topResult = results?.[0] ?? null;
  const quickActions = [
    'Tedavi protokolü nedir?',
    'Benzer vakaları göster',
    'Detaylı rapor oluştur',
  ];

  // Auto-scroll devre dışı - scroll manuel kontrol
  // useEffect(() => {
  //   if (!messagesContainerRef.current) return;

  //   // Yeni mesaj eklendiyse
  //   if (messages.length > prevMessagesLength.current) {
  //     const container = messagesContainerRef.current;
  //     const isNearBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 100;

  //     // Kullanıcı zaten en altta veya yakınındaysa, scroll yap
  //     if (isNearBottom || prevMessagesLength.current === 0) {
  //       messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  //     }
  //   }

  //   prevMessagesLength.current = messages.length;
  // }, [messages]);

  return (
    <Card className="h-full max-h-full flex flex-col animate-fade-in overflow-hidden">
      {/* Header */}
      {!analysisComplete || !topResult ? (
        <div className="p-4 border-b border-gray-200/50 flex-shrink-0">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center shadow-md">
              <FiMessageSquare className="text-white" size={18} />
            </div>
            <div>
              <h2 className="text-base font-semibold text-gray-900">AI Rapor & Asistan</h2>
              <p className="text-xs text-gray-600">Analiz sonuçları burada görünecek</p>
            </div>
          </div>
        </div>
      ) : (
        <StickyResultHeader topResult={topResult} />
      )}

      {/* Content - Basit max-height yaklaşımı */}
      <div
        ref={messagesContainerRef}
        className="p-4 space-y-4"
        style={{
          flex: '1 1 0',
          minHeight: 0,
          maxHeight: '100%',
          overflowY: 'auto',
          overflowX: 'hidden'
        }}
      >
        {/* Diagnosis Results - Her zaman göster */}
        {(results || loading) && (
          <div className="pb-4 border-b border-gray-200/50">
            <DiagnosisResults results={results} loading={loading} patientInfo={patientInfo} />
          </div>
        )}

        {/* Chat Messages - Analiz tamamlandığında göster */}
        {analysisComplete && messages.map((msg, index) => (
          <ChatMessage
            key={index}
            message={msg}
          />
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Footer */}
      {analysisComplete && (
        <div className="p-3 border-t border-gray-200/50 bg-white/50 flex-shrink-0">
          <ChatInput
            onSendMessage={onSendMessage}
            disabled={loading}
            quickActions={messages.length <= 1 ? quickActions : []}
          />
        </div>
      )}
    </Card>
  );
};

RightPanel.propTypes = {
  results: PropTypes.array,
  loading: PropTypes.bool,
  messages: PropTypes.array,
  onSendMessage: PropTypes.func.isRequired,
  analysisComplete: PropTypes.bool,
  patientInfo: PropTypes.shape({
    age: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
    gender: PropTypes.string,
    position: PropTypes.string,
  }),
};

export default RightPanel;

