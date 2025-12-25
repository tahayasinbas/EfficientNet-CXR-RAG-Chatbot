import React from 'react';
import PropTypes from 'prop-types';
import { FiMinus, FiPlus } from 'react-icons/fi';
import { PATIENT } from '../../constants';
import { Button } from '../ui/button';
import { Input } from '../ui/input';

const FormField = ({ label, htmlFor, children }) => (
  <div className="space-y-1.5">
    <label htmlFor={htmlFor} className="text-xs font-medium text-text-secondary">
      {label}
    </label>
    <div>{children}</div>
  </div>
);

const PatientForm = ({ patientData, setPatientData }) => {
  const handleChange = (field, value) => {
    // Yaş alanı için sayısal değerleri ve sınırı kontrol et
    if (field === 'age') {
      const numValue = value === '' ? '' : parseInt(value, 10);
      if (numValue === '' || (numValue >= PATIENT.AGE_MIN && numValue <= PATIENT.AGE_MAX)) {
        setPatientData((prev) => ({ ...prev, [field]: numValue }));
      }
    } else {
      setPatientData((prev) => ({ ...prev, [field]: value }));
    }
  };

  const handleAgeIncrement = (increment) => {
    const currentAge = patientData.age === '' ? 0 : parseInt(patientData.age, 10);
    const newAge = currentAge + increment;
    if (newAge >= PATIENT.AGE_MIN && newAge <= PATIENT.AGE_MAX) {
      handleChange('age', newAge);
    }
  };
  
  return (
    <div className="space-y-3">
      {/* Yaş */}
      <FormField label="Yaş" htmlFor="age-input">
        <div className="flex items-center gap-1.5">
          <Button
            type="button"
            variant="outline"
            size="icon"
            className="h-8 w-8 shrink-0"
            onClick={() => handleAgeIncrement(-1)}
            disabled={patientData.age <= PATIENT.AGE_MIN}
          >
            <FiMinus size={12} />
          </Button>
          <Input
            id="age-input"
            type="number"
            value={patientData.age}
            onChange={(e) => handleChange('age', e.target.value)}
            className="text-center h-8 text-sm"
            placeholder="58"
          />
          <Button
            type="button"
            variant="outline"
            size="icon"
            className="h-8 w-8 shrink-0"
            onClick={() => handleAgeIncrement(1)}
            disabled={patientData.age >= PATIENT.AGE_MAX}
          >
            <FiPlus size={12} />
          </Button>
        </div>
      </FormField>

      {/* Cinsiyet */}
      <FormField label="Cinsiyet">
        <div className="grid grid-cols-2 gap-1.5">
          <Button
            type="button"
            variant={patientData.gender === PATIENT.GENDERS.MALE ? 'default' : 'outline'}
            onClick={() => handleChange('gender', PATIENT.GENDERS.MALE)}
            className="h-8 text-xs"
          >
            Erkek
          </Button>
          <Button
            type="button"
            variant={patientData.gender === PATIENT.GENDERS.FEMALE ? 'default' : 'outline'}
            onClick={() => handleChange('gender', PATIENT.GENDERS.FEMALE)}
            className="h-8 text-xs"
          >
            Kadın
          </Button>
        </div>
      </FormField>

      {/* Pozisyon */}
      <FormField label="Pozisyon">
        <div className="grid grid-cols-2 gap-1.5">
          <Button
            type="button"
            variant={patientData.position === PATIENT.POSITIONS.PA ? 'default' : 'outline'}
            onClick={() => handleChange('position', PATIENT.POSITIONS.PA)}
            className="h-8 text-xs"
          >
            PA
          </Button>
          <Button
            type="button"
            variant={patientData.position === PATIENT.POSITIONS.AP ? 'default' : 'outline'}
            onClick={() => handleChange('position', PATIENT.POSITIONS.AP)}
            className="h-8 text-xs"
          >
            AP
          </Button>
        </div>
      </FormField>
    </div>
  );
};

PatientForm.propTypes = {
  patientData: PropTypes.object.isRequired,
  setPatientData: PropTypes.func.isRequired,
};

export default PatientForm;

