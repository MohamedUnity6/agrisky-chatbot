from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

class PlantHealthAnalyzer:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.plant_encoder = None
        
    def load_model(self, path='plant_health_model.pkl'):
        """Load trained model"""
        if os.path.exists(path):
            model_data = joblib.load(path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.plant_encoder = model_data['plant_encoder']
            return True
        return False
    
    def train(self, df):
        """Train model on the fly"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        import numpy as np
        
        self.plant_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
        # Prepare data
        conditions = []
        for _, row in df.iterrows():
            if row['Plant'] == 'Arugula':
                optimal = {
                    'Soil_Moisture': (50, 70),
                    'Temperature': (18, 22),
                    'Air_Humidity': (55, 65),
                    'pH': (6.2, 6.8),
                    'Light_Hours': (6, 8),
                    'EC': (1.2, 1.6)
                }
            else:
                optimal = {
                    'Soil_Moisture': (25, 35),
                    'Temperature': (24, 28),
                    'Air_Humidity': (48, 58),
                    'pH': (6.8, 7.4),
                    'Light_Hours': (8, 10),
                    'EC': (0.9, 1.3)
                }
            
            issues = []
            if row['Soil_Moisture'] < optimal['Soil_Moisture'][0]:
                issues.append('needs_water')
            elif row['Soil_Moisture'] > optimal['Soil_Moisture'][1]:
                issues.append('overwatered')
                
            if row['Light_Hours'] < optimal['Light_Hours'][0]:
                issues.append('needs_more_sun')
            elif row['Light_Hours'] > optimal['Light_Hours'][1]:
                issues.append('too_much_sun')
                
            if row['pH'] < optimal['pH'][0] or row['pH'] > optimal['pH'][1]:
                issues.append('soil_ph_issue')
                
            if row['EC'] < optimal['EC'][0]:
                issues.append('needs_fertilizer')
            elif row['EC'] > optimal['EC'][1]:
                issues.append('excess_fertilizer')
                
            if row['Temperature'] < optimal['Temperature'][0]:
                issues.append('too_cold')
            elif row['Temperature'] > optimal['Temperature'][1]:
                issues.append('too_hot')
                
            if row['Air_Humidity'] < optimal['Air_Humidity'][0]:
                issues.append('low_humidity')
            elif row['Air_Humidity'] > optimal['Air_Humidity'][1]:
                issues.append('high_humidity')
            
            if not issues:
                conditions.append('optimal')
            else:
                issues.sort()
                conditions.append(','.join(issues))
        
        df['Condition'] = conditions
        
        # Prepare features
        X = df[['Soil_Moisture', 'Temperature', 'Air_Humidity', 
                'pH', 'Light_Hours', 'EC']].values
        
        deviations = []
        for _, row in df.iterrows():
            if row['Plant'] == 'Arugula':
                optimal = {'Soil_Moisture': 60, 'Temperature': 20, 
                          'Air_Humidity': 60, 'pH': 6.5, 
                          'Light_Hours': 7, 'EC': 1.4}
            else:
                optimal = {'Soil_Moisture': 30, 'Temperature': 26, 
                          'Air_Humidity': 53, 'pH': 7.1, 
                          'Light_Hours': 9, 'EC': 1.1}
            
            dev = [
                abs(row['Soil_Moisture'] - optimal['Soil_Moisture']),
                abs(row['Temperature'] - optimal['Temperature']),
                abs(row['Air_Humidity'] - optimal['Air_Humidity']),
                abs(row['pH'] - optimal['pH']),
                abs(row['Light_Hours'] - optimal['Light_Hours']),
                abs(row['EC'] - optimal['EC'])
            ]
            deviations.append(dev)
        
        deviations = np.array(deviations)
        
        plant_encoded = self.plant_encoder.fit_transform(df['Plant'])
        plant_encoded = plant_encoded.reshape(-1, 1)
        
        X_combined = np.hstack([X, deviations, plant_encoded])
        X_scaled = self.scaler.fit_transform(X_combined)
        y = df['Condition']
        
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.model.fit(X_scaled, y)
        
        return self.model
    
    def predict(self, plant_type, soil_moisture, temperature, 
                air_humidity, ph, light_hours, ec):
        """Make prediction"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        try:
            plant_encoded = self.plant_encoder.transform([plant_type])[0]
        except:
            plant_encoded = 0
        
        if plant_type == 'Arugula':
            optimal = {'Soil_Moisture': 60, 'Temperature': 20, 
                      'Air_Humidity': 60, 'pH': 6.5, 
                      'Light_Hours': 7, 'EC': 1.4}
        else:
            optimal = {'Soil_Moisture': 30, 'Temperature': 26, 
                      'Air_Humidity': 53, 'pH': 7.1, 
                      'Light_Hours': 9, 'EC': 1.1}
        
        deviations = np.array([
            abs(soil_moisture - optimal['Soil_Moisture']),
            abs(temperature - optimal['Temperature']),
            abs(air_humidity - optimal['Air_Humidity']),
            abs(ph - optimal['pH']),
            abs(light_hours - optimal['Light_Hours']),
            abs(ec - optimal['EC'])
        ])
        
        input_features = np.array([[soil_moisture, temperature, air_humidity, 
                                    ph, light_hours, ec]])
        input_deviations = deviations.reshape(1, -1)
        input_plant = np.array([[plant_encoded]])
        
        input_data = np.hstack([input_features, input_deviations, input_plant])
        input_scaled = self.scaler.transform(input_data)
        
        prediction = self.model.predict(input_scaled)[0]
        probabilities = self.model.predict_proba(input_scaled)[0]
        confidence = max(probabilities) * 100
        
        recommendations = self._generate_recommendations(
            prediction, plant_type, soil_moisture, temperature,
            air_humidity, ph, light_hours, ec
        )
        
        # Also return detailed analysis
        analysis = self._get_detailed_analysis(
            plant_type, soil_moisture, temperature,
            air_humidity, ph, light_hours, ec
        )
        
        return {
            'status': 'healthy' if prediction == 'optimal' else 'needs_attention',
            'condition': prediction,
            'confidence': f"{confidence:.1f}%",
            'recommendations': recommendations,
            'detailed_analysis': analysis,
            'sensor_readings': {
                'plant_type': plant_type,
                'soil_moisture': soil_moisture,
                'temperature': temperature,
                'air_humidity': air_humidity,
                'ph': ph,
                'light_hours': light_hours,
                'ec': ec
            }
        }
    
    def _get_detailed_analysis(self, plant_type, soil_moisture, temperature, 
                               air_humidity, ph, light_hours, ec):
        """Get detailed analysis of each parameter"""
        if plant_type == 'Arugula':
            optimal = {
                'Soil_Moisture': (50, 70),
                'Temperature': (18, 22),
                'Air_Humidity': (55, 65),
                'pH': (6.2, 6.8),
                'Light_Hours': (6, 8),
                'EC': (1.2, 1.6)
            }
        else:
            optimal = {
                'Soil_Moisture': (25, 35),
                'Temperature': (24, 28),
                'Air_Humidity': (48, 58),
                'pH': (6.8, 7.4),
                'Light_Hours': (8, 10),
                'EC': (0.9, 1.3)
            }
        
        analysis = {}
        
        # Soil Moisture Analysis
        if soil_moisture < optimal['Soil_Moisture'][0]:
            analysis['soil_moisture'] = {
                'status': 'low',
                'current': soil_moisture,
                'optimal': f"{optimal['Soil_Moisture'][0]}-{optimal['Soil_Moisture'][1]}%",
                'message': f"Soil is too dry ({soil_moisture}%). Plants need more water."
            }
        elif soil_moisture > optimal['Soil_Moisture'][1]:
            analysis['soil_moisture'] = {
                'status': 'high',
                'current': soil_moisture,
                'optimal': f"{optimal['Soil_Moisture'][0]}-{optimal['Soil_Moisture'][1]}%",
                'message': f"Soil is too wet ({soil_moisture}%). Risk of root rot."
            }
        else:
            analysis['soil_moisture'] = {
                'status': 'good',
                'current': soil_moisture,
                'optimal': f"{optimal['Soil_Moisture'][0]}-{optimal['Soil_Moisture'][1]}%",
                'message': f"Soil moisture is optimal ({soil_moisture}%)."
            }
        
        # Temperature Analysis
        if temperature < optimal['Temperature'][0]:
            analysis['temperature'] = {
                'status': 'low',
                'current': temperature,
                'optimal': f"{optimal['Temperature'][0]}-{optimal['Temperature'][1]}°C",
                'message': f"Temperature is too cold ({temperature}°C). Growth may slow."
            }
        elif temperature > optimal['Temperature'][1]:
            analysis['temperature'] = {
                'status': 'high',
                'current': temperature,
                'optimal': f"{optimal['Temperature'][0]}-{optimal['Temperature'][1]}°C",
                'message': f"Temperature is too hot ({temperature}°C). Plant may wilt or bolt."
            }
        else:
            analysis['temperature'] = {
                'status': 'good',
                'current': temperature,
                'optimal': f"{optimal['Temperature'][0]}-{optimal['Temperature'][1]}°C",
                'message': f"Temperature is optimal ({temperature}°C)."
            }
        
        # pH Analysis
        if ph < optimal['pH'][0]:
            analysis['ph'] = {
                'status': 'low',
                'current': ph,
                'optimal': f"{optimal['pH'][0]}-{optimal['pH'][1]}",
                'message': f"Soil is too acidic (pH {ph}). Nutrient uptake may be affected."
            }
        elif ph > optimal['pH'][1]:
            analysis['ph'] = {
                'status': 'high',
                'current': ph,
                'optimal': f"{optimal['pH'][0]}-{optimal['pH'][1]}",
                'message': f"Soil is too alkaline (pH {ph}). Nutrient uptake may be affected."
            }
        else:
            analysis['ph'] = {
                'status': 'good',
                'current': ph,
                'optimal': f"{optimal['pH'][0]}-{optimal['pH'][1]}",
                'message': f"Soil pH is optimal ({ph})."
            }
        
        # Light Analysis
        if light_hours < optimal['Light_Hours'][0]:
            analysis['light'] = {
                'status': 'low',
                'current': light_hours,
                'optimal': f"{optimal['Light_Hours'][0]}-{optimal['Light_Hours'][1]} hours",
                'message': f"Not enough light ({light_hours}h/day). Plants need more sun."
            }
        elif light_hours > optimal['Light_Hours'][1]:
            analysis['light'] = {
                'status': 'high',
                'current': light_hours,
                'optimal': f"{optimal['Light_Hours'][0]}-{optimal['Light_Hours'][1]} hours",
                'message': f"Too much direct light ({light_hours}h/day). Consider partial shade."
            }
        else:
            analysis['light'] = {
                'status': 'good',
                'current': light_hours,
                'optimal': f"{optimal['Light_Hours'][0]}-{optimal['Light_Hours'][1]} hours",
                'message': f"Light is optimal ({light_hours}h/day)."
            }
        
        # EC/Nutrient Analysis
        if ec < optimal['EC'][0]:
            analysis['ec'] = {
                'status': 'low',
                'current': ec,
                'optimal': f"{optimal['EC'][0]}-{optimal['EC'][1]} mS/cm",
                'message': f"Nutrient level is low (EC {ec}). Plants need fertilizer."
            }
        elif ec > optimal['EC'][1]:
            analysis['ec'] = {
                'status': 'high',
                'current': ec,
                'optimal': f"{optimal['EC'][0]}-{optimal['EC'][1]} mS/cm",
                'message': f"Nutrient level is high (EC {ec}). Reduce fertilizer."
            }
        else:
            analysis['ec'] = {
                'status': 'good',
                'current': ec,
                'optimal': f"{optimal['EC'][0]}-{optimal['EC'][1]} mS/cm",
                'message': f"Nutrient level is optimal (EC {ec})."
            }
        
        return analysis
    
    def _generate_recommendations(self, condition, plant_type, soil_moisture, 
                                  temperature, air_humidity, ph, light_hours, ec):
        """Generate recommendations"""
        if condition == 'optimal':
            return [f"✅ Your {plant_type} is in optimal condition! Continue current care routine."]
        
        recommendations = []
        issues = condition.split(',')
        
        if plant_type == 'Arugula':
            optimal = {
                'Soil_Moisture': (50, 70),
                'Temperature': (18, 22),
                'Air_Humidity': (55, 65),
                'pH': (6.2, 6.8),
                'Light_Hours': (6, 8),
                'EC': (1.2, 1.6)
            }
        else:
            optimal = {
                'Soil_Moisture': (25, 35),
                'Temperature': (24, 28),
                'Air_Humidity': (48, 58),
                'pH': (6.8, 7.4),
                'Light_Hours': (8, 10),
                'EC': (0.9, 1.3)
            }
        
        for issue in set(issues):
            if issue == 'needs_water':
                recommendations.append(f"💧 **WATER**: Soil is {soil_moisture}% (needs {optimal['Soil_Moisture'][0]}-{optimal['Soil_Moisture'][1]}%). Water immediately and check daily.")
            elif issue == 'overwatered':
                recommendations.append(f"💧 **STOP WATERING**: Soil is {soil_moisture}% (should be {optimal['Soil_Moisture'][0]}-{optimal['Soil_Moisture'][1]}%). Let soil dry completely.")
            elif issue == 'needs_more_sun':
                recommendations.append(f"☀️ **MORE SUN**: Only {light_hours}h/day (needs {optimal['Light_Hours'][0]}-{optimal['Light_Hours'][1]}h). Move to sunnier spot.")
            elif issue == 'too_much_sun':
                recommendations.append(f"☀️ **LESS SUN**: {light_hours}h/day is too much. Provide shade during peak hours.")
            elif issue == 'soil_ph_issue':
                if ph < optimal['pH'][0]:
                    recommendations.append(f"🧪 **RAISE pH**: Current pH {ph} is too acidic. Add lime to reach {optimal['pH'][0]}-{optimal['pH'][1]}.")
                else:
                    recommendations.append(f"🧪 **LOWER pH**: Current pH {ph} is too alkaline. Add sulfur to reach {optimal['pH'][0]}-{optimal['pH'][1]}.")
            elif issue == 'needs_fertilizer':
                recommendations.append(f"🌱 **FERTILIZE**: EC {ec} is low (needs {optimal['EC'][0]}-{optimal['EC'][1]}). Apply balanced fertilizer.")
            elif issue == 'excess_fertilizer':
                recommendations.append(f"🌱 **STOP FERTILIZER**: EC {ec} is too high. Flush soil with water.")
            elif issue == 'too_cold':
                recommendations.append(f"🌡️ **WARM UP**: {temperature}°C is too cold (needs {optimal['Temperature'][0]}-{optimal['Temperature'][1]}°C). Move to warmer area.")
            elif issue == 'too_hot':
                recommendations.append(f"🌡️ **COOL DOWN**: {temperature}°C is too hot. Provide shade and ventilation.")
            elif issue == 'low_humidity':
                recommendations.append(f"💨 **MORE HUMIDITY**: {air_humidity}% is too dry. Mist leaves or use humidifier.")
            elif issue == 'high_humidity':
                recommendations.append(f"💨 **LESS HUMIDITY**: {air_humidity}% is too high. Improve air circulation.")
        
        return recommendations

# Initialize analyzer
analyzer = PlantHealthAnalyzer()

# Try to load existing model
if not analyzer.load_model():
    print("Training new model...")
    try:
        df = pd.read_csv('agrisky_sensors_dataset.csv')
        analyzer.train(df)
        print("Model trained successfully")
    except Exception as e:
        print(f"Error training model: {e}")

def extract_sensor_data_from_text(text):
    """Extract sensor data from natural language text"""
    text_lower = text.lower()
    
    # Extract plant type
    plant_type = None
    if 'arugula' in text_lower:
        plant_type = 'Arugula'
    elif 'thyme' in text_lower:
        plant_type = 'Thyme'
    
    # Extract numbers
    numbers = re.findall(r'\d+\.?\d*', text)
    
    if plant_type and len(numbers) >= 6:
        try:
            return {
                'plant_type': plant_type,
                'soil_moisture': float(numbers[0]),
                'temperature': float(numbers[1]),
                'air_humidity': float(numbers[2]),
                'ph': float(numbers[3]),
                'light_hours': float(numbers[4]),
                'ec': float(numbers[5])
            }
        except:
            pass
    return None

def get_plant_advice(plant_type, topic):
    """Get specific advice for plant care"""
    if plant_type == 'Arugula':
        if topic == 'yellow_leaves':
            return "🔍 **Yellow leaves on Arugula** usually indicate:\n• Overwatering (soil too wet)\n• Nutrient deficiency (low nitrogen)\n• Too much heat (above 25°C)\n\nShare your sensor data for accurate diagnosis!"
        elif topic == 'water':
            return "💧 **Arugula Watering**: Keep soil consistently moist (50-70%). Water when top inch feels dry. Avoid waterlogging."
        elif topic == 'sun':
            return "☀️ **Arugula Light**: Needs 6-8 hours of sunlight daily. Can tolerate partial shade but grows best in full sun."
        elif topic == 'temperature':
            return "🌡️ **Arugula Temperature**: Prefers cool weather (18-22°C). Will bolt (flower) quickly above 25°C."
        elif topic == 'fertilizer':
            return "🌱 **Arugula Fertilizer**: Feed every 2-3 weeks with balanced fertilizer (NPK 10-10-10). EC should be 1.2-1.6 mS/cm."
    else:  # Thyme
        if topic == 'yellow_leaves':
            return "🔍 **Yellow leaves on Thyme** usually indicate:\n• Overwatering (root rot)\n• Poor drainage\n• Nutrient deficiency\n\nShare your sensor data for accurate diagnosis!"
        elif topic == 'water':
            return "💧 **Thyme Watering**: Let soil dry completely between watering (25-35%). Thyme is drought-tolerant and hates wet feet."
        elif topic == 'sun':
            return "☀️ **Thyme Light**: Needs 8-10 hours of full sun. Grows best in bright, direct sunlight."
        elif topic == 'temperature':
            return "🌡️ **Thyme Temperature**: Thrives in warm conditions (24-28°C). Can tolerate heat but protect from frost."
        elif topic == 'fertilizer':
            return "🌱 **Thyme Fertilizer**: Light feeder - fertilize monthly with half-strength fertilizer. EC should be 0.9-1.3 mS/cm."

def diagnose_problem(problem_text):
    """Diagnose common plant problems"""
    problem_lower = problem_text.lower()
    
    if 'yellow' in problem_lower or 'yellowing' in problem_lower:
        return "🔍 **Yellow Leaves Diagnosis**:\n\nPossible causes:\n1. **Overwatering** - Soil too wet, roots suffocating\n2. **Underwatering** - Soil too dry, plant stressed\n3. **Nutrient deficiency** - Lack of nitrogen or iron\n4. **Poor drainage** - Roots sitting in water\n5. **Pests** - Aphids or spider mites\n\n📊 **To help diagnose, please share:**\n• Plant type (Arugula/Thyme)\n• Soil moisture level (%)\n• How often you water\n• Any other symptoms (wilting, spots, etc.)\n\nOr share your full sensor data for accurate analysis!"
    
    elif 'wilting' in problem_lower:
        return "🥀 **Wilting Diagnosis**:\n\nPossible causes:\n1. **Underwatering** - Most common cause\n2. **Overwatering** - Roots damaged by rot\n3. **Heat stress** - Too hot (check temperature)\n4. **Root damage** - From transplanting or pests\n\n💡 **Action**: Check soil moisture first. If dry (<40%), water immediately. If wet (>70%), stop watering and improve drainage."
    
    elif 'brown' in problem_lower or 'browning' in problem_lower:
        return "🤎 **Brown Leaves Diagnosis**:\n\nPossible causes:\n1. **Underwatering** - Leaves dry and crispy\n2. **Sunburn** - Too much direct sun\n3. **Fertilizer burn** - Too much fertilizer\n4. **Low humidity** - Air too dry\n\nShare your sensor data for specific recommendations!"
    
    elif 'spots' in problem_lower or 'spot' in problem_lower:
        return "🔴 **Leaf Spots Diagnosis**:\n\nPossible causes:\n1. **Fungal disease** - From wet leaves\n2. **Bacterial infection** - Poor air circulation\n3. **Pest damage** - Insects feeding\n4. **Nutrient burn** - Too much fertilizer\n\n**Action**: Remove affected leaves, improve air circulation, avoid wetting leaves when watering."
    
    elif 'pests' in problem_lower or 'bugs' in problem_lower:
        return "🐛 **Pest Problems**:\n\nCommon pests:\n• **Aphids** - Small green/black insects on new growth\n• **Spider mites** - Tiny webs on leaves\n• **Whiteflies** - Small white flies when disturbed\n\n**Treatment**: Spray with neem oil or insecticidal soap. Isolate affected plant."
    
    else:
        return "I can help diagnose plant problems! Tell me about:\n• Yellow leaves\n• Wilting\n• Brown spots\n• Pests\n\nOr share your sensor data for complete health analysis!"

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'online',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_plant():
    try:
        data = request.get_json()
        
        plant_type = data.get('plant_type', 'Arugula')
        soil_moisture = float(data.get('soil_moisture', 0))
        temperature = float(data.get('temperature', 0))
        air_humidity = float(data.get('air_humidity', 0))
        ph = float(data.get('ph', 0))
        light_hours = float(data.get('light_hours', 0))
        ec = float(data.get('ec', 0))
        
        result = analyzer.predict(
            plant_type, soil_moisture, temperature,
            air_humidity, ph, light_hours, ec
        )
        
        return jsonify({
            'success': True,
            'data': result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 400

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        message = data.get('message', '')
        user_id = data.get('user_id', 'anonymous')
        
        message_lower = message.lower()
        
        # Check if user is asking about problems without sensor data
        problem_keywords = ['yellow', 'wilting', 'brown', 'spot', 'pest', 'dying', 'problem', 'issue']
        if any(keyword in message_lower for keyword in problem_keywords):
            # Check if they also mentioned a plant type
            if 'arugula' in message_lower:
                plant = 'Arugula'
            elif 'thyme' in message_lower:
                plant = 'Thyme'
            else:
                plant = None
            
            # Give diagnostic advice
            response = diagnose_problem(message)
            if plant:
                response += f"\n\nFor {plant}, here's more specific advice:\n" + get_plant_advice(plant, 'yellow_leaves')
            
            return jsonify({
                'success': True,
                'response': response,
                'needs_sensor_data': True,
                'timestamp': datetime.now().isoformat()
            })
        
        # Check if they're asking about specific care
        if 'water' in message_lower:
            if 'arugula' in message_lower:
                response = get_plant_advice('Arugula', 'water')
            elif 'thyme' in message_lower:
                response = get_plant_advice('Thyme', 'water')
            else:
                response = "💧 **Watering Guide**:\n\n🌱 **Arugula**: Keep soil moist (50-70%). Water when top inch feels dry.\n🌿 **Thyme**: Let soil dry completely (25-35%). Water only when dry.\n\nWhich plant are you growing?"
            
            return jsonify({
                'success': True,
                'response': response,
                'timestamp': datetime.now().isoformat()
            })
        
        if 'sun' in message_lower or 'light' in message_lower:
            if 'arugula' in message_lower:
                response = get_plant_advice('Arugula', 'sun')
            elif 'thyme' in message_lower:
                response = get_plant_advice('Thyme', 'sun')
            else:
                response = "☀️ **Sunlight Guide**:\n\n🌱 **Arugula**: 6-8 hours of sunlight daily\n🌿 **Thyme**: 8-10 hours of full sun\n\nWhich plant are you growing?"
            
            return jsonify({
                'success': True,
                'response': response,
                'timestamp': datetime.now().isoformat()
            })
        
        if 'temperature' in message_lower or 'temp' in message_lower:
            if 'arugula' in message_lower:
                response = get_plant_advice('Arugula', 'temperature')
            elif 'thyme' in message_lower:
                response = get_plant_advice('Thyme', 'temperature')
            else:
                response = "🌡️ **Temperature Guide**:\n\n🌱 **Arugula**: 18-22°C (cool season)\n🌿 **Thyme**: 24-28°C (warm season)"
            
            return jsonify({
                'success': True,
                'response': response,
                'timestamp': datetime.now().isoformat()
            })
        
        if 'fertilizer' in message_lower or 'nutrient' in message_lower:
            if 'arugula' in message_lower:
                response = get_plant_advice('Arugula', 'fertilizer')
            elif 'thyme' in message_lower:
                response = get_plant_advice('Thyme', 'fertilizer')
            else:
                response = "🌱 **Fertilizer Guide**:\n\n🌱 **Arugula**: Every 2-3 weeks (EC 1.2-1.6)\n🌿 **Thyme**: Monthly, half-strength (EC 0.9-1.3)"
            
            return jsonify({
                'success': True,
                'response': response,
                'timestamp': datetime.now().isoformat()
            })
        
        # Check if they want optimal ranges
        if 'optimal' in message_lower or 'range' in message_lower or 'ideal' in message_lower:
            if 'arugula' in message_lower:
                plant = 'Arugula'
            elif 'thyme' in message_lower:
                plant = 'Thyme'
            else:
                plant = None
            
            if plant:
                response = f"📊 **Optimal Ranges for {plant}**:\n\n"
                response += f"💧 Soil Moisture: 50-70%\n"
                response += f"🌡️ Temperature: 18-22°C\n"
                response += f"💨 Air Humidity: 55-65%\n"
                response += f"🧪 pH: 6.2-6.8\n"
                response += f"☀️ Light: 6-8 hours\n"
                response += f"🌱 EC: 1.2-1.6 mS/cm"
            else:
                response = "📊 **Optimal Ranges**:\n\n**Arugula**:\n• Soil: 50-70% | Temp: 18-22°C | pH: 6.2-6.8\n• Light: 6-8h | EC: 1.2-1.6\n\n**Thyme**:\n• Soil: 25-35% | Temp: 24-28°C | pH: 6.8-7.4\n• Light: 8-10h | EC: 0.9-1.3"
            
            return jsonify({
                'success': True,
                'response': response,
                'timestamp': datetime.now().isoformat()
            })
        
        # Try to extract sensor data for analysis
        sensor_data = extract_sensor_data_from_text(message)
        
        if sensor_data:
            # Analyze with the sensor data
            result = analyzer.predict(
                sensor_data['plant_type'],
                sensor_data['soil_moisture'],
                sensor_data['temperature'],
                sensor_data['air_humidity'],
                sensor_data['ph'],
                sensor_data['light_hours'],
                sensor_data['ec']
            )
            
            # Format response
            if result['status'] == 'healthy':
                response = f"✅ **{sensor_data['plant_type']} is HEALTHY!**\n\n"
                response += f"📊 **Current Readings**:\n"
                response += f"• Soil Moisture: {sensor_data['soil_moisture']}% (Optimal: 50-70%)\n"
                response += f"• Temperature: {sensor_data['temperature']}°C (Optimal: 18-22°C)\n"
                response += f"• pH: {sensor_data['ph']} (Optimal: 6.2-6.8)\n"
                response += f"• Light: {sensor_data['light_hours']}h (Optimal: 6-8h)\n\n"
                response += f"💚 {result['recommendations'][0]}"
            else:
                response = f"⚠️ **{sensor_data['plant_type']} NEEDS ATTENTION!**\n\n"
                response += f"📊 **Detailed Analysis**:\n"
                
                # Add detailed analysis
                for param, analysis in result['detailed_analysis'].items():
                    if analysis['status'] != 'good':
                        response += f"{analysis['message']}\n"
                
                response += f"\n💡 **Recommendations**:\n"
                for rec in result['recommendations']:
                    response += f"{rec}\n"
            
            return jsonify({
                'success': True,
                'response': response,
                'analysis': result,
                'timestamp': datetime.now().isoformat()
            })
        
        # Greeting responses
        if any(word in message_lower for word in ['hello', 'hi', 'hey', 'greetings']):
            response = "🌱 Hello! I'm your plant health assistant. I can help you with:\n\n• **Analyze** your plant with sensor data\n• **Diagnose** problems like yellow leaves\n• **Advice** on watering, sunlight, fertilizer\n\nJust tell me what you need or share your sensor data!"
        
        # Help response
        elif 'help' in message_lower:
            response = "🤖 **I can help you with**:\n\n"
            response += "1️⃣ **Plant Analysis** - Share sensor data:\n"
            response += "   \"Arugula, soil 52, temp 21, humidity 60, ph 6.5, light 7, ec 1.3\"\n\n"
            response += "2️⃣ **Problem Diagnosis** - Ask about:\n"
            response += "   • \"Why are my arugula leaves yellow?\"\n"
            response += "   • \"My thyme is wilting\"\n\n"
            response += "3️⃣ **Care Advice** - Ask about:\n"
            response += "   • \"How often to water arugula?\"\n"
            response += "   • \"How much sun does thyme need?\"\n\n"
            response += "4️⃣ **Optimal Ranges** - Ask:\n"
            response += "   • \"What are optimal ranges for arugula?\""
        
        # Goodbye
        elif any(word in message_lower for word in ['bye', 'goodbye', 'thanks', 'thank you']):
            response = "🌿 You're welcome! Happy gardening! Come back if you need more help. Goodbye! 🌱"
        
        # Default response
        else:
            response = "I can help with plant care! Try asking about:\n\n• **Analyzing** your plant (share sensor data)\n• **Yellow leaves** diagnosis\n• **Watering** advice\n• **Sunlight** needs\n• **Optimal ranges**\n\nWhat would you like to know?"
        
        return jsonify({
            'success': True,
            'response': response,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 400

@app.route('/api/plants', methods=['GET'])
def get_plants():
    return jsonify({
        'success': True,
        'plants': ['Arugula', 'Thyme']
    })

@app.route('/api/optimal_ranges', methods=['GET'])
def get_optimal_ranges():
    plant = request.args.get('plant', 'Arugula')
    
    if plant == 'Arugula':
        ranges = {
            'plant_type': 'Arugula',
            'soil_moisture': {'min': 50, 'max': 70, 'unit': '%'},
            'temperature': {'min': 18, 'max': 22, 'unit': '°C'},
            'air_humidity': {'min': 55, 'max': 65, 'unit': '%'},
            'ph': {'min': 6.2, 'max': 6.8},
            'light_hours': {'min': 6, 'max': 8, 'unit': 'hours'},
            'ec': {'min': 1.2, 'max': 1.6, 'unit': 'mS/cm'}
        }
    else:
        ranges = {
            'plant_type': 'Thyme',
            'soil_moisture': {'min': 25, 'max': 35, 'unit': '%'},
            'temperature': {'min': 24, 'max': 28, 'unit': '°C'},
            'air_humidity': {'min': 48, 'max': 58, 'unit': '%'},
            'ph': {'min': 6.8, 'max': 7.4},
            'light_hours': {'min': 8, 'max': 10, 'unit': 'hours'},
            'ec': {'min': 0.9, 'max': 1.3, 'unit': 'mS/cm'}
        }
    
    return jsonify({
        'success': True,
        'data': ranges
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)