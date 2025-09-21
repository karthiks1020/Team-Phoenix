#!/usr/bin/env python3
"""
Quick Database Reset Script
"""
import sqlite3
import os

def reset_database():
    db_path = os.path.join(os.path.dirname(__file__), 'backend', 'artisans_hub.db')
    
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Delete all products
        cursor.execute("DELETE FROM product")
        conn.commit()
        
        # Count remaining data
        cursor.execute("SELECT COUNT(*) FROM seller")
        seller_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM product")
        product_count = cursor.fetchone()[0]
        
        conn.close()
        
        print(f"✅ Database reset complete!")
        print(f"📊 Sellers: {seller_count}, Products: {product_count}")
        print("🎆 Fresh marketplace state - sellers registered but no products uploaded!")
        
        return True
    else:
        print("❌ Database file not found")
        return False

if __name__ == "__main__":
    reset_database()