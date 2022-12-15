const { Ville } = require('../../db/sequelize')

  
module.exports = (app) => {
  app.post('/api/InsertVille', (req, res) => {

  
   Ville.create({
    code_iso2_ville:req.body.code_iso2_ville,
    nom_ville:req.body.nom_ville,
    PAYId:+req.body.PAYId,
    
  

    })
    return res.json('issertion Ville success')
    
   
   
  })
}