#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(__file__))

from ai_models.train_classifier import main

if __name__ == '__main__':
    print("🤖 Starting CNN Training Pipeline...")
    print("📊 This will train the handicraft classifier")
    main()
