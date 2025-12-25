from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator


class XRay(models.Model):
    """Model for storing X-Ray images and patient data"""

    GENDER_CHOICES = [
        ('M', 'Male'),
        ('F', 'Female'),
    ]

    POSITION_CHOICES = [
        ('PA', 'Posteroanterior'),
        ('AP', 'Anteroposterior'),
    ]

    # Patient Information
    age = models.IntegerField(
        validators=[MinValueValidator(0), MaxValueValidator(120)],
        help_text="Patient age (0-120)"
    )
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES)
    position = models.CharField(max_length=2, choices=POSITION_CHOICES)

    # Image
    image = models.ImageField(upload_to='xrays/%Y/%m/%d/')

    # Metadata
    uploaded_at = models.DateTimeField(auto_now_add=True)
    analyzed_at = models.DateTimeField(null=True, blank=True)

    # Analysis Status
    is_analyzed = models.BooleanField(default=False)

    class Meta:
        ordering = ['-uploaded_at']
        verbose_name = 'X-Ray'
        verbose_name_plural = 'X-Rays'

    def __str__(self):
        return f"XRay #{self.id} - {self.get_gender_display()}, {self.age}y"


class Diagnosis(models.Model):
    """Model for storing diagnosis results"""

    RISK_LEVELS = [
        ('low', 'Düşük'),
        ('medium', 'Orta'),
        ('high', 'Yüksek'),
    ]

    xray = models.ForeignKey(
        XRay,
        on_delete=models.CASCADE,
        related_name='diagnoses'
    )

    disease_name = models.CharField(max_length=100)
    confidence = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text="Prediction confidence (0.0-1.0)"
    )
    risk_level = models.CharField(max_length=10, choices=RISK_LEVELS)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-confidence']
        verbose_name = 'Diagnosis'
        verbose_name_plural = 'Diagnoses'

    def __str__(self):
        return f"{self.disease_name} ({self.confidence:.2%})"

    @property
    def percentage(self):
        """Return confidence as percentage"""
        return self.confidence * 100
