const { Pays } = require('../../db/sequelize')

  
module.exports = (app) => {
  app.post('/api/InsertPays', (req, res) => {

  
   Pays.create({
    code_iso2_pays:req.body.code_iso2_pays,
    nom_pays:req.body.nom_pays,
    DEVISEId:+req.body.DEVISEId
    
  

    })
    return res.json('issertion Pays success')
    
   
   
  })
}