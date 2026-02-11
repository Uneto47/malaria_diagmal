import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import hough_circle_peaks, hough_circle
from skimage.filters import sobel
from skimage.morphology import binary_opening, binary_closing, disk, binary_dilation
from skimage import color
from skimage.draw import circle_perimeter
from skimage.measure import label, regionprops

# ========== PARÂMETROS CONFIGURÁVEIS ==========
# Ajuste estes valores para melhorar a detecção

# Limiar de Binarização (multiplicador do máximo valor do Sobel)
# Teste valores entre 0.03 e 0.10
THRESHOLD_MULTIPLIER = 0.03  # Limiar = edges.max() * THRESHOLD_MULTIPLIER

# --- Parâmetros HSV para detecção de malária ---
# Canal Hue: identifica a tonalidade (roxo/azulado do parasita corado com Giemsa)
# Hue no skimage vai de 0.0 a 1.0 (0=vermelho, 0.17=amarelo, 0.33=verde, 0.5=ciano, 0.67=azul, 0.83=magenta)
HUE_MIN = 0.55          # Mínimo do hue (azul/roxo)
HUE_MAX = 0.95          # Máximo do hue (magenta/roxo)

# Canal Saturation: garante que a cor é saturada (não cinza/branco)
SAT_MIN = 0.15          # Saturação mínima (evita regiões acinzentadas)
SAT_MAX = 1.0           # Saturação máxima

# Canal Value: garante que não é muito claro (fundo) nem muito escuro (artefato)
VAL_MIN = 0.1           # Valor mínimo (evita regiões muito escuras/pretas)
VAL_MAX = 0.75          # Valor máximo (evita fundo claro e hemácias normais rosadas)

# Tamanho do disco para operações morfológicas na máscara HSV
HSV_MORPH_DISK_CLOSE = 5   # Disco para fechamento (une regiões próximas)
HSV_MORPH_DISK_OPEN = 3    # Disco para abertura (remove ruído)

# Área mínima de uma região de malária (em pixels) - regiões menores são descartadas
MIN_MALARIA_AREA = 200     # Ajuste conforme resolução da imagem

# Raios para detecção de Células Infectadas (em pixels)
MALARIA_RADIUS_MIN = 75  # Raio mínimo
MALARIA_RADIUS_MAX = 80  # Raio máximo
MALARIA_RADIUS_STEP = 2  # Incremento entre raios

# Raios para detecção de Hemácias Normais (em pixels)
RBC_RADIUS_MIN = 70  # Raio mínimo
RBC_RADIUS_MAX = 100  # Raio máximo (deve ser > MIN para np.arange funcionar)
RBC_RADIUS_STEP = 5  # Incremento entre raios

# Número máximo de círculos a detectar
MAX_MALARIA_CELLS = 50   # Máximo de células infectadas
MAX_RBC_CELLS = 200      # Máximo de hemácias normais

# Distância mínima entre círculos detectados (em pixels)
MIN_DISTANCE_MALARIA = 100  # Distância mínima entre células infectadas
MIN_DISTANCE_RBC = 100       # Distância mínima entre hemácias normais

# Raio de dilatação para mascarar células infectadas (em pixels)
MALARIA_MASK_DILATION = 40  # Raio maior para garantir remoção completa


def binarize_with_sobel(image, threshold_multiplier):
    """Binariza a imagem usando Sobel e limiar dinâmico"""
    edges = sobel(image)
    
    # Calcula o limiar dinamicamente
    limiar = edges.max() * threshold_multiplier
    
    # Binariza
    binary = edges.copy()
    binary[binary <= limiar] = 0
    binary[binary > limiar] = 1
    
    # Aplica operações morfológicas para remoção/redução de ruído
    binary = binary_closing(binary)
    binary = binary_opening(binary)
    
    return binary, edges

def remove_overlapping_circles(cx, cy, radii, accums, min_distance):
    """Remove círculos sobrepostos mantendo apenas o de maior acumulador"""
    if len(cx) == 0:
        return cx, cy, radii, accums
    
    # Ordena por acumulador (do maior para o menor)
    sorted_indices = np.argsort(accums)[::-1]
    
    keep_indices = []
    
    for i in sorted_indices:
        # Verifica se este círculo está muito próximo de algum já mantido
        keep = True
        for j in keep_indices:
            distance = np.sqrt((cx[i] - cx[j])**2 + (cy[i] - cy[j])**2)
            if distance < min_distance:
                keep = False
                break
        
        if keep:
            keep_indices.append(i)
    
    # Retorna apenas os círculos mantidos
    keep_indices = np.array(keep_indices)
    return cx[keep_indices], cy[keep_indices], radii[keep_indices], accums[keep_indices]

def process_malarian_cells(malarian_bin):
    """Processa a imagem binarizada para remover ruído e melhorar o reconhecimento das células infectadas"""
    edges = (sobel(malarian_bin)*255).astype("uint8")
    return malarian_bin, edges

def process_rb_cells(malarian_cells, all_cells_bin, image_gray):
    """Remove células de malária e processa a imagem de hemácias normais"""
    # Remove células de malária das células totais com dilatação maior
    mask = binary_dilation(malarian_cells, disk(MALARIA_MASK_DILATION))
    rb_cells = all_cells_bin.copy()
    rb_cells[mask > 0] = 0
    
    # Aplica Sobel na imagem em escala de cinza original, não na binária
    edges_gray = sobel(image_gray)
    edges = (edges_gray * 255).astype("uint8")
    # Mascara as bordas para remover regiões de malária
    edges[mask > 0] = 0
    
    return rb_cells, edges

def detect_malarian_cells(malarian_cells_edges, malarian_filled, original_img):
    """Detecta células infectadas por malária"""
    hough_radii = np.arange(MALARIA_RADIUS_MIN, MALARIA_RADIUS_MAX, MALARIA_RADIUS_STEP)
    hough_res = hough_circle(malarian_cells_edges, hough_radii)
    accums, cx, cy, radii = hough_circle_peaks(
        hough_res, hough_radii, 
        min_xdistance=MIN_DISTANCE_MALARIA, 
        min_ydistance=MIN_DISTANCE_MALARIA, 
        total_num_peaks=MAX_MALARIA_CELLS
    )
    
    # Remove círculos sobrepostos adicionais
    cx, cy, radii, accums = remove_overlapping_circles(cx, cy, radii, accums, MIN_DISTANCE_MALARIA)
    
    color_image = original_img.copy()
    circles_drawned = 0
    
    # Tamanho do X (em pixels)
    x_size = 30
    # Espessura do X (em pixels)
    thickness = 4
    # Cor azul turquesa chamativa (cyan vibrante)
    turquoise_color = (0, 255, 255)
    
    for center_y, center_x, radius in zip(cy, cx, radii):
        # Desenha um X azul turquesa grosso no centro da célula infectada
        # Linha diagonal 1 (\)
        for i in range(-x_size, x_size + 1):
            for offset_y in range(-thickness, thickness + 1):
                for offset_x in range(-thickness, thickness + 1):
                    y_coord = center_y + i + offset_y
                    x_coord = center_x + i + offset_x
                    if 0 <= y_coord < color_image.shape[0] and 0 <= x_coord < color_image.shape[1]:
                        color_image[y_coord, x_coord] = turquoise_color
        
        # Linha diagonal 2 (/)
        for i in range(-x_size, x_size + 1):
            for offset_y in range(-thickness, thickness + 1):
                for offset_x in range(-thickness, thickness + 1):
                    y_coord = center_y + i + offset_y
                    x_coord = center_x - i + offset_x
                    if 0 <= y_coord < color_image.shape[0] and 0 <= x_coord < color_image.shape[1]:
                        color_image[y_coord, x_coord] = turquoise_color
        
        circles_drawned += 1
    
    if circles_drawned == 0:
        print("Não foram encontradas células infectadas por malária nesta imagem!")
    else:
        print(f"Foram encontradas {circles_drawned} células infectadas por malária")
    
    # Retorna também as coordenadas dos círculos detectados
    return color_image, circles_drawned, hough_res, (cx, cy, radii)

def detect_rb_cells(rb_cells_edges, rb_cells_filled, malarian_detected, malaria_circles=None):
    """Detecta hemácias normais"""
    hough_radii = np.arange(RBC_RADIUS_MIN, RBC_RADIUS_MAX, RBC_RADIUS_STEP)
    hough_res = hough_circle(rb_cells_edges, hough_radii)
    accums, cx, cy, radii = hough_circle_peaks(
        hough_res, hough_radii, 
        min_xdistance=MIN_DISTANCE_RBC, 
        min_ydistance=MIN_DISTANCE_RBC, 
        total_num_peaks=MAX_RBC_CELLS
    )
    
    # Remove círculos sobrepostos
    cx, cy, radii, accums = remove_overlapping_circles(cx, cy, radii, accums, MIN_DISTANCE_RBC)
    
    # Remove círculos de RBC que se sobrepõem com células de malária
    if malaria_circles is not None:
        mal_cx, mal_cy, mal_radii = malaria_circles
        keep_indices = []
        
        for i in range(len(cx)):
            # Verifica se este círculo está muito próximo de alguma célula de malária
            keep = True
            for j in range(len(mal_cx)):
                distance = np.sqrt((cx[i] - mal_cx[j])**2 + (cy[i] - mal_cy[j])**2)
                # Se a distância for menor que a soma dos raios + margem, descarta
                if distance < (radii[i] + mal_radii[j] + 20):
                    keep = False
                    break
            
            if keep:
                keep_indices.append(i)
        
        # Filtra apenas os círculos mantidos
        if len(keep_indices) > 0:
            keep_indices = np.array(keep_indices)
            cx = cx[keep_indices]
            cy = cy[keep_indices]
            radii = radii[keep_indices]
    
    color_image = malarian_detected.copy()
    circles_drawned = 0
    
    for center_y, center_x, radius in zip(cy, cx, radii):
        # Desenha círculos verdes nas hemácias normais
        for r_offset in [-1, 0, 1]:
            circy, circx = circle_perimeter(center_y, center_x, radius + r_offset, shape=color_image.shape)
            color_image[circy, circx] = (20, 220, 20)
        circles_drawned += 1
    
    if circles_drawned == 0:
        print("Não foram encontradas hemácias normais nesta imagem!")
    else:
        print(f"Foram encontradas {circles_drawned} hemácias normais")
    
    return color_image, hough_res

def create_malaria_mask_hsv(image_rgb, binary_cells):
    """
    Cria uma máscara para células infectadas usando os 3 canais do HSV.
    
    Parasitas corados com Giemsa aparecem como manchas roxas/azuladas escuras.
    Hemácias normais são rosadas/avermelhadas e mais claras.
    
    Retorna:
        malaria_mask: máscara booleana das regiões de malária
        debug_info: dicionário com máscaras intermediárias para visualização
    """
    image_hsv = color.rgb2hsv(image_rgb)
    hue = image_hsv[:, :, 0]
    sat = image_hsv[:, :, 1]
    val = image_hsv[:, :, 2]
    
    # --- Máscara baseada nos 3 canais ---
    # Hue: seleciona tonalidades roxas/azuladas (parasita)
    # Trata o caso de hue_min > hue_max (quando a faixa cruza o 0/1)
    if HUE_MIN <= HUE_MAX:
        hue_mask = (hue >= HUE_MIN) & (hue <= HUE_MAX)
    else:
        # Faixa que cruza a fronteira 0/1 (ex: 0.85 a 0.15)
        hue_mask = (hue >= HUE_MIN) | (hue <= HUE_MAX)
    
    # Saturação: garante que há cor real (não é branco/cinza)
    sat_mask = (sat >= SAT_MIN) & (sat <= SAT_MAX)
    
    # Valor: exclui regiões muito claras (fundo) e muito escuras (artefatos)
    val_mask = (val >= VAL_MIN) & (val <= VAL_MAX)
    
    # Combina as 3 condições
    combined_mask = hue_mask & sat_mask & val_mask
    
    # Intersecciona com a binarização (só considerar onde há células)
    combined_mask = combined_mask & binary_cells
    
    # --- Operações morfológicas para limpar a máscara ---
    # Fechamento: une pequenos buracos e regiões fragmentadas
    combined_mask = binary_closing(combined_mask, disk(HSV_MORPH_DISK_CLOSE))
    # Abertura: remove ruído pequeno
    combined_mask = binary_opening(combined_mask, disk(HSV_MORPH_DISK_OPEN))
    
    # --- Remove regiões muito pequenas (ruído residual) ---
    labeled_mask = label(combined_mask)
    cleaned_mask = np.zeros_like(combined_mask, dtype=bool)
    for region in regionprops(labeled_mask):
        if region.area >= MIN_MALARIA_AREA:
            cleaned_mask[labeled_mask == region.label] = True
    
    # Informações de debug para visualização
    debug_info = {
        'hue': hue,
        'sat': sat,
        'val': val,
        'hue_mask': hue_mask,
        'sat_mask': sat_mask,
        'val_mask': val_mask,
        'combined_raw': hue_mask & sat_mask & val_mask,
        'combined_cleaned': cleaned_mask
    }
    
    return cleaned_mask, debug_info
# ==============================================================================================================================================================================
# ==============================================================================================================================================================================
# ==============================================================================================================================================================================


# Carrega a imagem
# original_img = imread('DSCN1365_JPG.rf.116e18e1c8a0201f1bec8005ab1db0f1.jpg')
original_img = imread('images/DSCN1365.JPG')
image_gray = color.rgb2gray(original_img)

# Binariza usando Sobel com limiar dinâmico
binary_all_cells, edges_sobel = binarize_with_sobel(image_gray, THRESHOLD_MULTIPLIER)

# Separa células de malária usando os 3 canais do HSV
malaria_mask, hsv_debug = create_malaria_mask_hsv(original_img, binary_all_cells)
malaria_binary = binary_all_cells.copy()
malaria_binary[~malaria_mask] = 0

# Processa células de malária
final_malarian_cells, final_malarian_cells_edges = process_malarian_cells(malaria_binary)

# Processa hemácias normais (removendo as células infectadas)
final_blood_cells, final_blood_cells_edges = process_rb_cells(final_malarian_cells, binary_all_cells, image_gray)

# Detecta células infectadas por malária (círculos vermelhos)
malarian_detected, num_malaria, malaria_hough, malaria_circles = detect_malarian_cells(final_malarian_cells_edges, final_malarian_cells, original_img)

# Detecta hemácias normais (círculos verdes) - passa as coordenadas das células de malária
final_result, rbc_hough = detect_rb_cells(final_blood_cells_edges, final_blood_cells, malarian_detected, malaria_circles)

# Histograma das bordas para análise
edge_histogram = np.histogram(edges_sobel.ravel(), bins=256)

# Visualização dos resultados
fig, axes = plt.subplots(5, 3, figsize=(15, 25))

# Linha 1 - Processamento inicial
axes[0, 0].imshow(original_img)
axes[0, 0].set_title('Imagem Original')
axes[0, 0].axis('off')

axes[0, 1].imshow(image_gray, cmap='gray')
axes[0, 1].set_title('Imagem em Escala de Cinza')
axes[0, 1].axis('off')

axes[0, 2].imshow(edges_sobel, cmap='gray')
axes[0, 2].set_title(f'Sobel da Imagem\n(limiar={THRESHOLD_MULTIPLIER}*max)')
axes[0, 2].axis('off')

# Linha 2 - Canais HSV (para calibração dos parâmetros)
axes[1, 0].imshow(hsv_debug['hue'], cmap='hsv')
axes[1, 0].set_title(f'Canal Hue\n(filtro: {HUE_MIN}–{HUE_MAX})')
axes[1, 0].axis('off')

axes[1, 1].imshow(hsv_debug['sat'], cmap='gray')
axes[1, 1].set_title(f'Canal Saturação\n(filtro: {SAT_MIN}–{SAT_MAX})')
axes[1, 1].axis('off')

axes[1, 2].imshow(hsv_debug['val'], cmap='gray')
axes[1, 2].set_title(f'Canal Valor\n(filtro: {VAL_MIN}–{VAL_MAX})')
axes[1, 2].axis('off')

# Linha 3 - Máscaras HSV e resultado combinado
axes[2, 0].imshow(hsv_debug['hue_mask'], cmap='gray')
axes[2, 0].set_title('Máscara Hue')
axes[2, 0].axis('off')

axes[2, 1].imshow(hsv_debug['combined_raw'], cmap='gray')
axes[2, 1].set_title('Máscara HSV Combinada (bruta)')
axes[2, 1].axis('off')

axes[2, 2].imshow(hsv_debug['combined_cleaned'], cmap='gray')
axes[2, 2].set_title('Máscara HSV Final (limpa)')
axes[2, 2].axis('off')

# Linha 4 - Detecção
axes[3, 0].imshow(final_malarian_cells, cmap='gray')
axes[3, 0].set_title(f'Células Infectadas\n(raios: {MALARIA_RADIUS_MIN}-{MALARIA_RADIUS_MAX})')
axes[3, 0].axis('off')

axes[3, 1].imshow(final_blood_cells, cmap='gray')
axes[3, 1].set_title(f'Hemácias Normais\n(raios: {RBC_RADIUS_MIN}-{RBC_RADIUS_MAX})')
axes[3, 1].axis('off')

axes[3, 2].imshow(final_result)
axes[3, 2].set_title(f'Detecção Final\n(Turquesa: {num_malaria} Infectadas | Verde: Normais)')
axes[3, 2].axis('off')

# Linha 5 - Análise dos Acumuladores de Hough
axes[4, 0].plot(edge_histogram[0], '-k')
axes[4, 0].set_title('Histograma das Bordas (Sobel)')
axes[4, 0].set_xlabel('Intensidade')
axes[4, 0].set_ylabel('Frequência')
axes[4, 0].grid(True, alpha=0.3)

if len(malaria_hough) > 0:
    malaria_accum_mean = np.mean(malaria_hough, axis=0)
    axes[4, 1].imshow(malaria_accum_mean, cmap='hot')
    axes[4, 1].set_title(f'Acumulador Hough - Malária\n(média dos raios {MALARIA_RADIUS_MIN}-{MALARIA_RADIUS_MAX})')
    axes[4, 1].axis('off')
else:
    axes[4, 1].text(0.5, 0.5, 'Sem dados', ha='center', va='center')
    axes[4, 1].axis('off')

if len(rbc_hough) > 0:
    rbc_accum_mean = np.mean(rbc_hough, axis=0)
    axes[4, 2].imshow(rbc_accum_mean, cmap='hot')
    axes[4, 2].set_title(f'Acumulador Hough - Hemácias\n(média dos raios {RBC_RADIUS_MIN}-{RBC_RADIUS_MAX})')
    axes[4, 2].axis('off')
else:
    axes[4, 2].text(0.5, 0.5, 'Sem dados', ha='center', va='center')
    axes[4, 2].axis('off')

plt.tight_layout()
plt.show()
