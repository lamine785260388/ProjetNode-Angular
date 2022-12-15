const { Agence } = require('../../db/sequelize')
  
module.exports = (app) => {
  app.put('/api/updateAgence/:id', (req, res) => {
    const id = req.params.id
    Agence.findByPk(id).then(agence => {
      if(agence===null){
        const message='l_agence que vous voulez modifier n_existe pas'
        return res.status(404).json({message})
      }
      Agence.update(req.body, {
        where: { id: id }
      })
      .then(_=>{
        message=`l_Agence ${agence.nom_agence} a bien été modifie le voici aprés la modification`
        Agence.findByPk(id).then(newagence=>{
          return res.status(201).json({message,data:newagence})
        })
       
      })
    })
    .catch(error=>{
      message='le serveur n_est pas disponible veuillez ressayer apres'
      res.status(501).json({message,data:error})
    })
  })
}