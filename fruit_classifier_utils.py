"""\nUtilidades para el Clasificador de Frutas y Verduras\n\nContiene clases para:\n- Extracción de características de imágenes (color, textura, forma)\n- Aumento de datos (data augmentation)\n"""

import numpy as np
import cv2


class FeatureExtractor:
    """\n    Extrae múltiples tipos de características de imágenes para clasificación.\n    \n    Características extraídas:\n    - Color: medias, desviaciones, histogramas RGB\n    - Textura: bordes Sobel, Laplacian\n    - Forma: área, perímetro, circularidad, aspect ratio\n    - Estadísticas: media, std, mediana, skewness\n    """
    
    def extract_color_features(self, img):
        """
        Extrae características basadas en el color de la imagen.
        
        Args:
            img: Imagen RGB normalizada (valores en [0,1])
        
        Returns:
            numpy.ndarray: Vector de 30 características de color
        """
        features = []
        
        # Media y desviación estándar por canal RGB
        for i in range(3):
            features.append(np.mean(img[:, :, i]))
            features.append(np.std(img[:, :, i]))
        
        # Histogramas de color (8 bins por canal)
        for i in range(3):
            hist, _ = np.histogram(img[:, :, i], bins=8, range=(0, 1))
            hist = hist / hist.sum()  # Normalizar
            features.extend(hist)
        
        return np.array(features)
    
    def extract_texture_features(self, img):
        """
        Extrae características de textura usando detección de bordes.
        
        Returns:
            numpy.ndarray: Vector de 6 características de textura
        """
        # Convertir a escala de grises
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        features = []
        
        # Bordes Sobel (horizontal y vertical)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        features.append(np.mean(np.abs(sobelx)))
        features.append(np.std(np.abs(sobelx)))
        features.append(np.mean(np.abs(sobely)))
        features.append(np.std(np.abs(sobely)))
        
        # Filtro Laplacian para detección de bordes
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        features.append(np.mean(np.abs(laplacian)))
        features.append(np.std(np.abs(laplacian)))
        
        return np.array(features)
    
    def extract_shape_features(self, img):
        """
        Extrae características basadas en la forma del objeto.
        
        Returns:
            numpy.ndarray: Vector de 4 características de forma
                          (área, perímetro, circularidad, aspect ratio)
        """
        features = []
        
        # Convertir a escala de grises para umbralizado
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Umbralizado binario automático (Otsu)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            # Obtener el contorno más grande
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Área normalizada
            area = cv2.contourArea(largest_contour)
            features.append(area / (img.shape[0] * img.shape[1]))
            
            # Perímetro normalizado
            perimeter = cv2.arcLength(largest_contour, True)
            features.append(perimeter / (2 * (img.shape[0] + img.shape[1])))
            
            # Circularidad (qué tan circular es el objeto: 1=círculo perfecto)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
                features.append(circularity)
            else:
                features.append(0)
            
            # Aspect ratio (relación ancho/alto)
            x, y, w, h = cv2.boundingRect(largest_contour)
            if h > 0:
                aspect_ratio = w / h
                features.append(aspect_ratio)
            else:
                features.append(1)
        else:
            # No se encontraron contornos
            features.extend([0, 0, 0, 1])
        
        return np.array(features)
    
    def extract_statistical_features(self, img):
        """Extract statistical features from the image"""
        features = []
        
        # Flatten image to 1D
        flat = img.reshape(-1, img.shape[2])
        
        # Overall statistics
        features.append(np.mean(flat))
        features.append(np.std(flat))
        features.append(np.median(flat))
        features.append(np.min(flat))
        features.append(np.max(flat))
        
        # Skewness approximation
        mean = np.mean(flat)
        std = np.std(flat)
        if std > 0:
            skewness = np.mean(((flat - mean) / std) ** 3)
            features.append(skewness)
        else:
            features.append(0)
        
        return np.array(features)
    
    def extract_all_features(self, img):
        """
        Extrae todas las características de una imagen.
        
        Args:
            img: Imagen RGB normalizada (valores en [0,1])
        
        Returns:
            numpy.ndarray: Vector combinado de ~46 características
        """
        color_features = self.extract_color_features(img)
        texture_features = self.extract_texture_features(img)
        shape_features = self.extract_shape_features(img)
        statistical_features = self.extract_statistical_features(img)
        
        # Concatenar todas las características en un solo vector
        all_features = np.concatenate([
            color_features,
            texture_features,
            shape_features,
            statistical_features
        ])
        
        return all_features


class DataAugmentation:
    """\n    Técnicas de aumento de datos para expandir el dataset de entrenamiento.\n    \n    Transformaciones disponibles:\n    - Geométricas: rotación, flip, crop, zoom\n    - Color: brillo, contraste, saturación\n    - Ruido: gaussiano aleatorio\n    """
    
    def rotate(self, img, angle=None):
        """Rota la imagen un ángulo aleatorio entre -30° y 30°"""
        if angle is None:
            angle = np.random.uniform(-30, 30)
        
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        return rotated
    
    def flip_horizontal(self, img):
        """Voltea la imagen horizontalmente (espejo)"""
        return cv2.flip(img, 1)
    
    def flip_vertical(self, img):
        """Voltea la imagen verticalmente"""
        return cv2.flip(img, 0)
    
    def adjust_brightness(self, img, factor=None):
        """Ajusta el brillo de la imagen (factor 0.7-1.3)"""
        if factor is None:
            factor = np.random.uniform(0.7, 1.3)
        
        adjusted = img * factor
        return np.clip(adjusted, 0, 1)
    
    def adjust_contrast(self, img, factor=None):
        """Ajusta el contraste de la imagen"""
        if factor is None:
            factor = np.random.uniform(0.7, 1.3)
        
        mean = np.mean(img)
        adjusted = (img - mean) * factor + mean
        return np.clip(adjusted, 0, 1)
    
    def adjust_saturation(self, img, factor=None):
        """Ajusta la saturación de color de la imagen"""
        if factor is None:
            factor = np.random.uniform(0.7, 1.3)
        
        # Convertir a HSV para modificar saturación
        hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # Ajustar canal de saturación
        hsv[:, :, 1] = hsv[:, :, 1] * factor
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        
        # Convertir de vuelta a RGB
        rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return rgb.astype(np.float32) / 255.0
    
    def add_noise(self, img, noise_factor=None):
        """Agrega ruido gaussiano aleatorio a la imagen"""
        if noise_factor is None:
            noise_factor = np.random.uniform(0.01, 0.05)
        
        noise = np.random.randn(*img.shape) * noise_factor
        noisy = img + noise
        return np.clip(noisy, 0, 1)
    
    def random_crop_and_resize(self, img, crop_factor=None):
        """Recorta aleatoriamente y redimensiona al tamaño original"""
        if crop_factor is None:
            crop_factor = np.random.uniform(0.8, 0.95)
        
        h, w = img.shape[:2]
        
        new_h = int(h * crop_factor)
        new_w = int(w * crop_factor)
        
        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)
        
        cropped = img[top:top + new_h, left:left + new_w]
        resized = cv2.resize(cropped, (w, h))
        
        return resized
    
    def zoom(self, img, zoom_factor=None):
        """Zoom in/out on image"""
        if zoom_factor is None:
            zoom_factor = np.random.uniform(0.9, 1.1)
        
        h, w = img.shape[:2]
        
        # Resize image
        new_h = int(h * zoom_factor)
        new_w = int(w * zoom_factor)
        resized = cv2.resize(img, (new_w, new_h))
        
        # Crop or pad to original size
        if zoom_factor > 1:
            # Crop from center
            top = (new_h - h) // 2
            left = (new_w - w) // 2
            result = resized[top:top + h, left:left + w]
        else:
            # Pad to center
            result = np.zeros_like(img)
            top = (h - new_h) // 2
            left = (w - new_w) // 2
            result[top:top + new_h, left:left + new_w] = resized
        
        return result
