const { Client } = require('../../db/sequelize')

  
module.exports = (app) => {
  app.post('/api/InsertClient', async (req, res) => {
    

  
    const [user, created] =  await Client.findOrCreate({
      where: { id: req.body.id },
      defaults: {
        id:req.body.id,
        nom_client:req.body.nom_client,
       prenom_client:req.body.prenom_client,
       phone:req.body.phone
      }
    });
    if(created){
      return res.json({data:user.id});
    }
    else{
      return res.json({data:user.id})
    }
    
   
   
  })
}