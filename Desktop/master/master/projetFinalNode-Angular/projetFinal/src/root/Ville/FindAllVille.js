
const { Ville } = require('../../db/sequelize')

  
module.exports = (app) => {
  app.get('/api/findAllVille', (req, res) => {
  
  
      Ville.findAll({order:['nom_ville' ]})
      .then(Ville => {
        const message = 'La liste des Villes a bien été récupérée.'
        res.json({ message, data: Ville })
      })
     .catch(error=>{
      const message="la liste des Villes n'est pas disponible.Résseyer dans quelques instant!"
      res.status(500).json({message,data:error})
     })
    
   
  })
}