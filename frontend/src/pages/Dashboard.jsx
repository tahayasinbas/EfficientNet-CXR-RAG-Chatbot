import React, { useState } from 'react';
import MainLayout from '../components/Layout/MainLayout';
import LeftPanel from '../components/LeftPanel/LeftPanel';
import CenterPanel from '../components/CenterPanel/CenterPanel';
import RightPanel from '../components/RightPanel/RightPanel';
import { uploadXRay, analyzeXRay, sendChatMessage } from '../services/api';

const Dashboard = () => {
  // State
  const [selectedImage, setSelectedImage] = useState(null);
  const [patientData, setPatientData] = useState({
    age: '',
    gender: 'M',
    position: 'PA',
  });
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [heatmapImage, setHeatmapImage] = useState(null);

  // Chat State
  const [messages, setMessages] = useState([]);
  const [analysisComplete, setAnalysisComplete] = useState(false);
  const [chatSessionId, setChatSessionId] = useState(null);
  const [currentXRayId, setCurrentXRayId] = useState(null);

  // Panel State
  const [isLeftPanelCollapsed, setIsLeftPanelCollapsed] = useState(false);

  // Analiz fonksiyonu
  const handleAnalyze = async () => {
    if (!selectedImage) return;

    setLoading(true);
    setResults(null);
    setHeatmapImage(null);
    setMessages([]);
    setAnalysisComplete(false);

    try {
      // 1. Görüntüyü yükle
      const formData = new FormData();
      formData.append('image', selectedImage.file);
      formData.append('age', patientData.age || '');
      formData.append('gender', patientData.gender);
      formData.append('position', patientData.position);

      const uploadResponse = await uploadXRay(formData);
      const xrayId = uploadResponse.data.id;
      setCurrentXRayId(xrayId);

      // 2. Analiz et
      const analyzeResponse = await analyzeXRay(xrayId);

      // 3. Sonuçları set et - Backend'den gelen gerçek veriler
      const predictions = analyzeResponse.data?.xray?.diagnoses || [];
      setResults(predictions);
      setAnalysisComplete(true);

      // 4. AI'nin otomatik ilk mesajını ekle
      const topDisease = predictions[0];
      const initialAIMessage = {
        content: `Görüntüyü analiz ettim. EfficientNet-B3 modeline göre %${(topDisease.confidence * 100).toFixed(0)} olasılıkla ${topDisease.disease_name} bulgusu saptandı. Sormak istediğiniz bir şey var mı?`,
        sender: 'ai',
        timestamp: Date.now(),
        id: Date.now()
      };

      setMessages([initialAIMessage]);

      // Mock heatmap (gerçek backend'den gelecek)
      // setHeatmapImage('data:image/png;base64,...');

    } catch (error) {
      console.error('Analiz hatası:', error);
      alert('Analiz sırasında bir hata oluştu. Lütfen tekrar deneyin.');
    } finally {
      setLoading(false);
    }
  };

  // Kullanıcı mesajı gönderme
  const handleSendMessage = async (userMessage) => {
    // Kullanıcı mesajını ekle
    const newUserMessage = {
      content: userMessage,
      sender: 'user',
      timestamp: Date.now(),
      id: Date.now()
    };

    setMessages((prev) => [...prev, newUserMessage]);

    try {
      // Backend'e mesaj gönder
      const response = await sendChatMessage({
        sessionId: chatSessionId,
        xrayId: currentXRayId,
        message: userMessage
      });

      // Session ID'yi sakla
      if (!chatSessionId && response.data.session_id) {
        setChatSessionId(response.data.session_id);
      }

      // AI yanıtını ekle
      const aiMessage = {
        content: response.data.ai_message.content,
        sender: 'ai',
        timestamp: Date.now(),
        id: response.data.ai_message.id,
        source: response.data.ai_message.rag_source
      };

      setMessages((prev) => [...prev, aiMessage]);

    } catch (error) {
      console.error('Mesaj gönderme hatası:', error);

      // Hata durumunda fallback mesajı
      const errorMessage = {
        content: 'Üzgünüm, şu anda yanıt veremiyorum. Lütfen tekrar deneyin.',
        sender: 'ai',
        timestamp: Date.now(),
        id: Date.now()
      };

      setMessages((prev) => [...prev, errorMessage]);
    }
  };

  return (
    <MainLayout>
      {/* 3 Sütunlu Grid Yapısı - Dinamik */}
      <div className="grid grid-cols-12 gap-4 h-full">
        {/* Sol Panel - Collapsed: 1, Expanded: 3 */}
        <div className={isLeftPanelCollapsed ? "col-span-1" : "col-span-3"}>
          <LeftPanel
            selectedImage={selectedImage}
            onImageSelect={setSelectedImage}
            patientData={patientData}
            setPatientData={setPatientData}
            onAnalyze={handleAnalyze}
            loading={loading}
            isCollapsed={isLeftPanelCollapsed}
            onToggleCollapse={setIsLeftPanelCollapsed}
          />
        </div>

        {/* Orta Panel - Dinamik genişlik */}
        <div className={isLeftPanelCollapsed ? "col-span-6" : "col-span-5"}>
          <CenterPanel
            image={selectedImage?.preview}
            heatmapImage={heatmapImage}
          />
        </div>

        {/* Sağ Panel - Dinamik genişlik */}
        <div className={isLeftPanelCollapsed ? "col-span-5" : "col-span-4"}>
          <RightPanel
            results={results}
            loading={loading}
            messages={messages}
            onSendMessage={handleSendMessage}
            analysisComplete={analysisComplete}
            patientInfo={patientData}
          />
        </div>
      </div>
    </MainLayout>
  );
};

export default Dashboard;
