const express = require('express');
const multer = require('multer');
const cors = require('cors');
const pdf = require('pdf-parse');
const { PDFDocument } = require('pdf-lib');
const OpenAI = require('openai');
const fs = require('fs').promises;
const path = require('path');
const pdfjsLib = require('pdfjs-dist/legacy/build/pdf');
const { createCanvas } = require('canvas');
require('dotenv').config();

const app = express();
const port = process.env.PORT || 5000;
pdfjsLib.GlobalWorkerOptions.workerSrc = require.resolve('pdfjs-dist/legacy/build/pdf.worker.js');

// Configure OpenAI
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, 'uploads/');
  },
  filename: (req, file, cb) => {
    cb(null, Date.now() + '-' + file.originalname);
  }
});

const upload = multer({ 
  storage,
  fileFilter: (req, file, cb) => {
    if (file.mimetype === 'application/pdf') {
      cb(null, true);
    } else {
      cb(new Error('Only PDF files are allowed'));
    }
  },
  limits: { fileSize: 10 * 1024 * 1024 } // 10MB limit
});

// Middleware
// app.use(cors());
app.use(cors({
  origin: [
    'https://nutri-scan-frontend.vercel.app',  // Your actual frontend URL
    'http://localhost:3000'  // Keep for local development
  ]
}));
app.use(express.json());

// Create uploads directory if it doesn't exist
const initUploadsDir = async () => {
  try {
    await fs.mkdir('uploads', { recursive: true });
  } catch (error) {
    console.error('Error creating uploads directory:', error);
  }
};

initUploadsDir();

// Function to extract text from PDF
async function extractTextFromPDF(filePath) {
  try {
    const dataBuffer = await fs.readFile(filePath);
    const data = await pdf(dataBuffer);
    return data.text;
  } catch (error) {
    console.error('Error extracting text from PDF:', error);
    throw error;
  }
}

// Function to check if PDF is scanned (image-based)
async function isPDFScanned(filePath) {
  try {
    const text = await extractTextFromPDF(filePath);
    // If text length is very small, it's likely a scanned PDF
    return text.trim().length < 50;
  } catch (error) {
    return true; // Assume scanned if can't extract text
  }
}

// Function to convert ALL pages of PDF to images
async function convertPDFToImages(filePath) {
  try {
    const dataBuffer = await fs.readFile(filePath);
    const loadingTask = pdfjsLib.getDocument({
      data: new Uint8Array(dataBuffer),
      useSystemFonts: true
    });
    
    const pdfDocument = await loadingTask.promise;
    const numPages = pdfDocument.numPages;
    const images = [];
    
    console.log(`Converting ${numPages} page(s) to images...`);
    
    // Process all pages
    for (let pageNum = 1; pageNum <= numPages; pageNum++) {
      const page = await pdfDocument.getPage(pageNum);
      
      const viewport = page.getViewport({ scale: 3.0 }); // Higher scale = better quality for OCR
      const canvas = createCanvas(viewport.width, viewport.height);
      const context = canvas.getContext('2d');
      
      const renderContext = {
        canvasContext: context,
        viewport: viewport
      };
      
      await page.render(renderContext).promise;
      
      // Convert canvas to base64 PNG
      const base64Image = canvas.toBuffer('image/png').toString('base64');
      images.push(base64Image);
      console.log(`Converted page ${pageNum}/${numPages}`);
    }
    
    return images; // Return array of all page images
  } catch (error) {
    console.error('Error converting PDF to images:', error);
    throw error;
  }
}

// Function to extract structured data using OpenAI from text
async function extractDataWithAI(text) {
  try {
    const prompt = `You are an expert at extracting allergen and nutritional information from food product descriptions. The text may be in Hungarian or other languages.

Look for these Hungarian terms:
- Allergének/Allergén anyagok (Allergens)
- Tápértékek/Tápérték (Nutritional values)
- Energia (Energy) - may show as kcal or kJ
- Zsír/Zsírtartalom (Fat)
- Szénhidrát (Carbohydrate)
- Cukor (Sugar)
- Fehérje (Protein)
- Só/Nátrium (Salt/Sodium)

Extract the following information from the text and return it in valid JSON format:

{
  "allergens": {
    "gluten": boolean,
    "egg": boolean,
    "crustaceans": boolean,
    "fish": boolean,
    "peanut": boolean,
    "soy": boolean,
    "milk": boolean,
    "tree_nuts": boolean,
    "celery": boolean,
    "mustard": boolean
  },
  "nutritional_values": {
    "energy": "value with unit (e.g., 250 kcal or 1046 kJ)",
    "fat": "value with unit (e.g., 10g)",
    "carbohydrate": "value with unit (e.g., 30g)",
    "sugar": "value with unit (e.g., 5g)",
    "protein": "value with unit (e.g., 8g)",
    "sodium": "value with unit (e.g., 0.5g)"
  }
}

Rules:
- For allergens, set to true if present/detected, false if explicitly stated as absent, null if not mentioned
- For nutritional values, include the value with its unit. Use null if not found
- Look for variations like "contains", "may contain", "traces of" for allergens
- Be thorough in checking tables, lists, and paragraphs
- If the information is not clearly stated, use null

Text to analyze:
${text}`;

    const response = await openai.chat.completions.create({
      model: "gpt-4o",
      messages: [
        {
          role: "system",
          content: "You are a helpful assistant that extracts allergen and nutritional information from food product descriptions. Always respond with valid JSON."
        },
        {
          role: "user",
          content: prompt
        }
      ],
      temperature: 0.1,
      response_format: { type: "json_object" }
    });

    const result = JSON.parse(response.choices[0].message.content);
    return result;
  } catch (error) {
    if (error.code === 'insufficient_quota' || error.status === 429) {
      console.warn('⚠️ OpenAI quota exceeded. Returning fallback response.');
      return {
        error: "OpenAI quota exceeded. Please check your billing or try again later.",
        fallback: true,
        allergens: {},
        nutritional_values: {}
      };
    }

    console.error('Error extracting data with AI:', error);
    throw error;
  }
}

// Function to extract data from scanned PDF using Vision API (ALL PAGES)
async function extractDataFromScannedPDF(filePath) {
  try {
    // Convert ALL PDF pages to PNG images
    const base64Images = await convertPDFToImages(filePath);
    
    console.log(`Processing ${base64Images.length} page(s) with Vision API...`);
    
    // Build content array with all pages
    const content = [
      {
        type: "text",
        text: `Extract allergen and nutritional information from this food product label. The text may be in Hungarian or other languages.

Look for these Hungarian terms:
- Allergének/Allergén anyagok (Allergens)
- Tápértékek/Tápérték (Nutritional values)
- Energia (Energy) - may show as kcal or kJ
- Zsír/Zsírtartalom (Fat)
- Szénhidrát (Carbohydrate)
- Cukor (Sugar)
- Fehérje (Protein)
- Só/Nátrium (Salt/Sodium)
- Glutén (Gluten), Tojás (Egg), Tej (Milk), Szója (Soy), etc.

Return data in this JSON format:

{
  "allergens": {
    "gluten": boolean,
    "egg": boolean,
    "crustaceans": boolean,
    "fish": boolean,
    "peanut": boolean,
    "soy": boolean,
    "milk": boolean,
    "tree_nuts": boolean,
    "celery": boolean,
    "mustard": boolean
  },
  "nutritional_values": {
    "energy": "value with unit (e.g., 250 kcal or 1046 kJ)",
    "fat": "value with unit (e.g., 10g)",
    "carbohydrate": "value with unit (e.g., 30g)",
    "sugar": "value with unit (e.g., 5g)",
    "protein": "value with unit (e.g., 8g)",
    "sodium": "value with unit (e.g., 0.5g or 500mg)"
  }
}

IMPORTANT: Look through ALL pages/images provided below. Look very carefully at nutritional tables, even if text is small. Use null only if truly not visible in any of the images.`
      }
    ];
    
    // Add all page images to the content
    base64Images.forEach((base64Image, index) => {
      content.push({
        type: "image_url",
        image_url: {
          url: `data:image/png;base64,${base64Image}`,
          detail: "high"
        }
      });
      console.log(`Added page ${index + 1} to API request`);
    });

    const response = await openai.chat.completions.create({
      model: "gpt-4o",
      messages: [
        {
          role: "system",
          content: "You are a helpful assistant that extracts allergen and nutritional information from food product images. Always respond with valid JSON."
        },
        {
          role: "user",
          content: content
        }
      ],
      temperature: 0.1,
      max_tokens: 1500,
      response_format: { type: "json_object" }
    });

    const result = JSON.parse(response.choices[0].message.content);
    console.log('Extraction complete!');
    return result;
  } catch (error) {
    if (error.code === 'insufficient_quota' || error.status === 429) {
      console.warn('⚠️ OpenAI quota exceeded. Returning fallback response.');
      return {
        error: "OpenAI quota exceeded. Please check your billing or try again later.",
        fallback: true,
        allergens: {},
        nutritional_values: {}
      };
    }
    
    console.error('Error extracting data from scanned PDF:', error);
    throw error;
  }
}

// Main endpoint for processing PDF
app.post('/api/extract', upload.single('pdf'), async (req, res) => {
  let filePath = null;
  
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No PDF file uploaded' });
    }

    filePath = req.file.path;
    console.log('Processing file:', filePath);

    // Check if PDF is scanned
    const isScanned = await isPDFScanned(filePath);
    console.log('Is scanned PDF:', isScanned);

    let extractedData;

    if (isScanned) {
      // Use Vision API for scanned PDFs
      extractedData = await extractDataFromScannedPDF(filePath);
    } else {
      // Extract text and use GPT for text-based PDFs
      const text = await extractTextFromPDF(filePath);
      extractedData = await extractDataWithAI(text);
    }

    // Clean up uploaded file
    await fs.unlink(filePath);

    res.json({
      success: true,
      data: extractedData,
      filename: req.file.originalname
    });

  } catch (error) {
    console.error('Error processing PDF:', error);
    
    // Clean up file if exists
    if (filePath) {
      try {
        await fs.unlink(filePath);
      } catch (unlinkError) {
        console.error('Error deleting file:', unlinkError);
      }
    }

    res.status(500).json({ 
      success: false,
      error: 'Failed to process PDF',
      details: error.message 
    });
  }
});

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', message: 'Server is running' });
});

app.listen(port, () => {
  console.log(`Server running on port ${port}`);
  console.log(`OpenAI API Key configured: ${!!process.env.OPENAI_API_KEY}`);
});