"""
gemini_service.py
-----------------
Integrates Google Gemini API for human-readable anomaly explanations
and energy-saving recommendations.

Important:
- Gemini is used ONLY for natural-language explanation.
- The ML model performs all predictions. Gemini does NOT predict energy.
- If the GEMINI_API_KEY environment variable is not set, the application
  still runs normally but returns a placeholder explanation.
- Gemini is instructed to clearly communicate that its explanations are
  possible interpretations, not verified physical diagnoses.
"""

import os
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load the Gemini API key from the environment (never hard-coded)
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


def _build_prompt(anomaly_info: dict) -> str:
    """
    Build a structured prompt for Gemini from anomaly observation data.

    Parameters
    ----------
    anomaly_info : dict
        Keys expected:
            actual_energy, predicted_energy, residual, severity,
            hour, day_of_week, T1 (indoor temp), T_out (outdoor temp),
            RH_1 (indoor humidity), RH_out (outdoor humidity),
            Windspeed, Visibility, lights, Tdewpoint
            (and optionally others)

    Returns
    -------
    str
        Formatted prompt text.
    """
    actual = anomaly_info.get("actual_energy", "N/A")
    predicted = anomaly_info.get("predicted_energy", "N/A")
    residual = anomaly_info.get("residual", "N/A")
    severity = anomaly_info.get("severity", "Unknown")
    hour = anomaly_info.get("hour", "N/A")
    dow = anomaly_info.get("day_of_week", "N/A")

    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_str = day_names[int(dow)] if isinstance(dow, (int, float)) and 0 <= int(dow) <= 6 else str(dow)

    prompt = f"""You are an AI energy advisor integrated into EcoPulse AI, an energy consumption monitoring system.

An anomaly has been detected in appliance energy consumption. Below is the observation data from sensors and the ML model.

--- OBSERVATION DATA ---
Actual appliance energy consumption: {actual} Wh
Model-predicted energy consumption:  {predicted} Wh
Residual (actual - predicted):       {residual} Wh
Anomaly severity:                    {severity}
Time of observation:                 {hour}:00, {day_str}

Environmental context:
- Indoor temperature (T1):          {anomaly_info.get('T1', 'N/A')} °C
- Outdoor temperature (T_out):      {anomaly_info.get('T_out', 'N/A')} °C
- Indoor humidity (RH_1):           {anomaly_info.get('RH_1', 'N/A')} %
- Outdoor humidity (RH_out):        {anomaly_info.get('RH_out', 'N/A')} %
- Wind speed:                       {anomaly_info.get('Windspeed', 'N/A')} m/s
- Visibility:                       {anomaly_info.get('Visibility', 'N/A')} km
- Dew point temperature:            {anomaly_info.get('Tdewpoint', 'N/A')} °C
- Lights energy consumption:        {anomaly_info.get('lights', 'N/A')} Wh

--- YOUR TASK ---
Please provide:
1. SHORT EXPLANATION (2-3 sentences): Why might consumption be unusually {'high' if float(residual if residual != 'N/A' else 0) > 0 else 'low'} compared to the model's expectation? Base your answer ONLY on the data provided above.

2. POSSIBLE CONTRIBUTING FACTORS (2-4 bullet points): List possible environmental or behavioural factors. You MUST state clearly that these are possible contributing factors, not verified physical causes.

3. ENERGY-SAVING RECOMMENDATIONS (2-4 bullet points): Practical recommendations relevant to this time, temperature, and consumption pattern.

4. SUSTAINABILITY IMPACT (1 sentence): Brief statement on the environmental significance of this anomaly.

IMPORTANT CONSTRAINTS:
- Do NOT invent sensor values not provided above.
- Do NOT claim certainty about physical causes.
- Always frame factors as possibilities.
- Keep the total response under 250 words.
- Format with clear section headers.
"""
    return prompt


def get_gemini_explanation(anomaly_info: dict) -> str:
    """
    Query Gemini for a human-readable anomaly explanation.

    Parameters
    ----------
    anomaly_info : dict
        Structured anomaly data (see _build_prompt for keys).

    Returns
    -------
    str
        Gemini's explanation and recommendations, or a placeholder if the
        API key is unavailable.
    """
    if not GEMINI_API_KEY:
        return (
            "⚠️ Gemini API key not configured. "
            "Set the GEMINI_API_KEY environment variable to enable "
            "AI-powered explanations and energy-saving recommendations. "
            "See .env.example for setup instructions."
        )

    try:
        import google.generativeai as genai

        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = _build_prompt(anomaly_info)
        response = model.generate_content(prompt)
        return response.text.strip()

    except ImportError:
        logger.error("google-generativeai package not installed.")
        return (
            "⚠️ Gemini integration requires the 'google-generativeai' package. "
            "Run: pip install google-generativeai"
        )
    except Exception as exc:
        logger.error("Gemini API call failed: %s", exc)
        return f"⚠️ Gemini API error: {exc}\nPlease check your API key and network connection."
