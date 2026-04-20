# movierec
movie reccomendation algorithm


**Srikar Chitturi,**
**Sukhleen Dhadiala,**
**Arsh Wasnik,**
**Aiden Zemblaku, and**
**Sophia Ziegler**

**Repo Structure**

```
movierec/
├── backend/
│   ├── server.py           
│   └── requirements.txt    # Python dependencies
├── frontend/              
├── model/
│   └── knn.py              # KNN recommendation model
├── ml-100k/                # MovieLens 100K dataset
│   ├── u.data              # User ratings
│   ├── u.item              # Movie metadata
│   ├── u.user              # User demographics
│   └── ...                 # more files etc
├── simulation/
│   ├── simulation.py       # Main simulation script
│   └── initialsimulation.py
├── webapp/                 # Frontend
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── MovieCard.jsx
│   │   │   ├── ResultCard.jsx
│   │   │   └── StarRating.jsx
│   │   └── ...
│   ├── public/
│   ├── package.json
│   └── vite.config.js
├── main.ipynb              # Jupyter notebook (exploration/development)
└── scratch.py
```

**How to start up the webpage!**


1. Check if Node is installed, run:
    - node -v
       - if it returns that the command wasn't found, install node

2. Go to the LTS version of Node.js from the official site and make sure that you add Node to PATH during installation.

3. Fully close terminal and reopen it, then run:
   - node -v
   - npm - v
     - This should give you the version numbers

4. To start the webapp first direct yourself into the webapp directory and run:
   - npm install
   - npm run dev
     - This will output something like this:
       - Local:     http://localhost:5000/
       - Network:   use --host to expose
       - press h + enter to show help

5. In your browser, input the link:
   - http://localhost:####/ with the specific numbers (#) in place

**Congratulations! The front end should be working now! All that's left is connecting the backend.**

6. First, open a new terminal window and it's important to keep the first one running in the background.
   - You'll need two terminal windows opened and running in order for the website to work.

7. Direct yourself into the backend directory, and run these lines:
    - py (or python3 or python) -m pip install -r requirements.txt
    - py (or python3 or python) server.py
      - This will install the required packages for the backend to function

8. Refresh the web page and the movierec website will be functional!
   - This is the method that worked for me and my device, it might be slightly different for your specific device.
