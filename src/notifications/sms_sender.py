import os
from twilio.rest import Client
from datetime import datetime, timedelta
import time

# Twilio configuration (you'll need to set these as environment variables)
# TWILIO_ACCOUNT_SID=your_account_sid
# TWILIO_AUTH_TOKEN=your_auth_token
# TWILIO_PHONE_NUMBER=your_twilio_phone_number
# ADMIN_PHONE_NUMBERS=comma_separated_phone_numbers (e.g., +1234567890,+1987654321)

# Dictionary to track notifications with timestamps and additional context
notification_timestamps = {}
# 5 minutes cooldown between notifications for the same event type
NOTIFICATION_COOLDOWN_MINUTES = 5  

# Dictionary to track the last notification details for each event type
last_notification_details = {}

def send_sms_alert(event_type, message, image_url=None, context=None):
    """
    Send an SMS alert to all admin numbers
    
    Args:
        event_type (str): Type of event ('unknown_face' or 'dangerous_item')
        message (str): The alert message to send
        image_url (str, optional): URL of the image if available
        context (dict, optional): Additional context about the event for better deduplication
    """
    try:
        # Create a unique key for this specific event
        event_key = event_type
        if context:
            # Include relevant context in the key to make it more specific
            if event_type == 'dangerous_item' and 'item_name' in context:
                event_key = f"{event_type}_{context['item_name'].lower()}"
            elif event_type == 'unknown_face' and 'location' in context:
                event_key = f"{event_type}_{context['location'].lower().replace(' ', '_')}"
        
        # Skip if we've sent a similar notification recently
        if not should_send_notification(event_key):
            print(f"⏳ Skipping SMS notification for {event_key} - sent recently")
            return False
            
        # Get Twilio credentials from environment variables
        account_sid = os.getenv('TWILIO_ACCOUNT_SID')
        auth_token = os.getenv('TWILIO_AUTH_TOKEN')
        twilio_phone = os.getenv('TWILIO_PHONE_NUMBER')
        admin_numbers = os.getenv('ADMIN_PHONE_NUMBERS', '').split(',')
        
        if not all([account_sid, auth_token, twilio_phone]) or not admin_numbers:
            print("❌ Missing Twilio configuration. Please set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER, and ADMIN_PHONE_NUMBERS")
            return False
            
        # Initialize Twilio client
        client = Client(account_sid, auth_token)
        
        # Prepare the message
        full_message = f"🚨 SECURITY ALERT - {event_type.upper()}\n\n{message}"
        if image_url:
            full_message += f"\n\nImage: {image_url}"
        
        # Send to all admin numbers
        for phone_number in admin_numbers:
            phone_number = phone_number.strip()
            if not phone_number:
                continue
                
            try:
                message = client.messages.create(
                    body=full_message,
                    from_=twilio_phone,
                    to=phone_number
                )
                print(f"📱 SMS alert sent to {phone_number}")
                
            except Exception as e:
                print(f"❌ Error sending SMS to {phone_number}: {str(e)}")
        
        # Record this notification with the specific event key
        record_notification(event_key)
        last_notification_details[event_key] = {
            'timestamp': datetime.now(),
            'event_type': event_type,
            'context': context or {}
        }
        return True
        
    except Exception as e:
        print(f"❌ Error in send_sms_alert: {str(e)}")
        return False

def record_notification(event_type):
    """Record that a notification was sent for this event type"""
    notification_timestamps[event_type] = datetime.now()

def should_send_notification(event_type):
    """Check if we should send a notification for this event type"""
    last_notification = notification_timestamps.get(event_type)
    if not last_notification:
        return True
        
    time_since_last = datetime.now() - last_notification
    return time_since_last > timedelta(minutes=NOTIFICATION_COOLDOWN_MINUTES)

def send_unknown_face_alert(location="main entrance", image_url=None):
    """
    Send an alert for an unknown face detection
    
    Args:
        location (str): Location where the face was detected
        image_url (str, optional): URL of the image if available
    """
    message = f"⚠️ Unknown person detected at {location} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    context = {
        'location': location,
        'detection_time': datetime.now().isoformat()
    }
    return send_sms_alert('unknown_face', message, image_url, context)

def send_dangerous_item_alert(item_name, confidence, location="campus", image_url=None):
    """
    Send an alert for a dangerous item detection
    
    Args:
        item_name (str): Name of the detected dangerous item
        confidence (float): Detection confidence (0-100)
        location (str): Location where the item was detected
        image_url (str, optional): URL of the image if available
    """
    message = (
        f"⚠️ POTENTIAL THREAT DETECTED!\n"
        f"Item: {item_name.upper()}\n"
        f"Confidence: {confidence:.1f}%\n"
        f"Location: {location}\n"
        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    context = {
        'item_name': item_name,
        'confidence': confidence,
        'location': location,
        'detection_time': datetime.now().isoformat()
    }
    return send_sms_alert('dangerous_item', message, image_url, context)
