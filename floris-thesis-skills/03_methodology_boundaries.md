The thesis studies the upper tail of hourly maximum wind gusts, denoted FX.

Primary objects:
- Upper-tail dependence between KNMI station locations 
- Pairwise extremal dependence 
- Conditional exceedance probabilities 
- Dependence decay with geographical distance  
- Seasonal comparison, especially winter versus summer  

Possible methods:
- Block maxima with max-stable process motivation
- Pairwise extremal coefficients
- Threshold exceedance-based tail dependence checks
- Robustness to block choice and threshold choice

# Method priority

Preferred main route:
1. Construct seasonal block maxima of FX by station.
2. Estimate pairwise extremal dependence between stations.
3. Relate dependence estimates to geographical distance.
4. Compare winter and summer dependence curves or summaries.
5. Use sample splitting only as secondary analysis.

Do not start with overly complex max-stable likelihood unless the simpler empirical dependence analysis is already clear.

Use max-stable processes primarily as theoretical motivation unless implementation time allows full model fitting.

Do not broaden the analysis to both wind speed and wind gusts unless explicitly requested. The main thesis variable is FX.

Important references:
- Smith (1990), max-stable processes and spatial extremes
- Padoan, Ribatet and Sisson (2010), likelihood-based inference for max-stable processes
- Davison, Padoan and Ribatet (2012), statistical modeling of spatial extremes