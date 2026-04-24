import os
import matplotlib.pyplot as plt
import numpy as np

# Configuration du style premium (Zéro Bullshit)
plt.style.use('dark_background')
plt.rcParams.update({
    "axes.facecolor": "#0B0F19",
    "figure.facecolor": "#0B0F19",
    "grid.color": "#232B3E",
    "text.color": "#E2E8F0",
    "axes.labelcolor": "#E2E8F0",
    "xtick.color": "#94A3B8",
    "ytick.color": "#94A3B8"
})

def generate_energy_chart(output_path="Image_5_Frugalite_Energie.png"):
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    
    # Données issues du Test 8 (CodeCarbon)
    models = ['Random Forest\n(Modèle Classique)', 'Matrice-256 / FaNet\n(IA Frugale)']
    energy_consumption = [125, 40] # en nano-Watts-heure (nWh) par inférence
    colors = ['#F43F5E', '#10B981'] # Rouge vs Vert néon
    
    # Tracer les barres
    bars = ax.bar(models, energy_consumption, color=colors, width=0.5, edgecolor='#232B3E', linewidth=2)
    
    # Ajout des valeurs sur les barres
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 3, f"{yval} nWh", 
                ha='center', va='bottom', fontsize=14, fontweight='bold', color='white')

    # Ligne de démarcation pour visualiser la réduction
    reduction_line_y = 40
    ax.axhline(y=reduction_line_y, color='#10B981', linestyle='--', linewidth=1.5, alpha=0.5)
    
    # Flèche pour indiquer la réduction massive
    ax.annotate(
        "Réduction de 67.8%\nDéployable sur\nPanneaux Solaires",
        xy=(0, 80), xycoords='data',
        xytext=(0.5, 80), textcoords='data',
        arrowprops=dict(arrowstyle="<->", color="#38BDF8", lw=2),
        ha='center', va='center', fontsize=12, color='#38BDF8', fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.5", fc="#151B2B", ec="#38BDF8", lw=1)
    )

    # Personnalisation
    ax.set_title("TEST 8 : Empreinte Carbone et Frugalité (Edge AI)", 
                 fontsize=16, color='#38BDF8', pad=20, fontweight='bold')
    ax.set_ylabel("Consommation Énergétique par Inférence (nWh)", fontsize=12, labelpad=10)
    ax.set_ylim([0, 140])
    
    # Grille subtile
    ax.grid(axis='y', linestyle=':', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#232B3E')
    ax.spines['bottom'].set_color('#232B3E')

    # Sauvegarde en haute définition
    plt.tight_layout()
    plt.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    generate_energy_chart()
