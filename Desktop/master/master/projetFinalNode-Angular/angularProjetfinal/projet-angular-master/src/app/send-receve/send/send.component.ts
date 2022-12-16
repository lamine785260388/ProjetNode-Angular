import { Router } from '@angular/router';
import { OnInit } from '@angular/core';
import { Component } from '@angular/core';
import { NgForm } from '@angular/forms';
import { Service } from 'src/app/node.service';
import { HttpClient, HttpHeaders } from "@angular/common/http";
import { Pays } from 'src/app/class/pays';
import Swal from 'sweetalert2';
import { findonePays_Devices } from 'src/app/class/PaysDeviseFindOne';
import { Devise } from 'src/app/class/devise';
import { AllServicesService } from 'src/app/all-services.service';
import { Idclass } from 'src/app/class/id';

@Component({
  selector: 'app-send',
  templateUrl: './send.component.html',
  styleUrls: ['./send.component.css']
})
export class SendComponent implements OnInit {

constructor(private router:Router,private http: HttpClient,private mesServices:AllServicesService){
 if(sessionStorage.getItem('isloggin')!='true'){
  sessionStorage.setItem('url','send')
  this.router.navigate(['login'])
 }
 }
 idEmetteur: Idclass|any;
 idRecepteur: Idclass|any;
 verif:string="false"
 allPays:Pays[]|any
 infoem:findonePays_Devices|any
 inforec:findonePays_Devices|any
 infoemdev:Devise|any
 inforecdev:Devise|any

 httpOptions = {
  headers: new HttpHeaders({
    "Content-Type": "application/json"
  })
};
  ngOnInit(): void {

    console.log(sessionStorage.getItem('isloggin'))
    this.http
    .get<Pays[]|any>(
      "http://localhost:3000/api/findAllPays",
      )
      .subscribe(res=>{
this.allPays=res.data
console.log(this.allPays[0])
      })
   
   
  }
  
  submit (form: NgForm) {
     //recuperation des informations de l'emmeteur
    var prenomEm = form.value.prenomemetteur;
    var nomEm=form.value.nomemetteur;
    var cniEm=form.value.cniemetteur
    var phoneEm=form.value.phoneemetteur
//find or create emetteur(client)
    this.http
      .post<Idclass|any>(
        "http://localhost:3000/api/InsertClient",
        { id: cniEm, nom_client: nomEm,prenom_client:prenomEm,phone:phoneEm },
        this.httpOptions
      )
      .subscribe(res=>{
        this.idEmetteur=res.data
        
        
      })
      //recupération information recepteur
      
      var prenom = form.value.prenomrecepteur;
    var nom=form.value.nomrecepteur;
    var cni=form.value.cnirecepteur;
    var phone=form.value.phonerecepteur;
    var montantenvoye=+form.value.montantenvoie;
    var montantreçu=montantenvoye-montantenvoye*0.01
    var idPaysemetteur=+form.value.idpaysemetteur
    var idPaysRecepteur=+form.value.idpaysrecepteur
    console.log('Emetteur'+idPaysemetteur)
    console.log('Recepteur'+idPaysRecepteur)
    //find or create recepteur(client)
    this.http
      .post<any>(
        "http://localhost:3000/api/InsertClient",
        { id: cni, nom_client: nom,prenom_client:prenom,phone:phone },
        this.httpOptions
      )
      .subscribe(res=>{
       this.idRecepteur=res.data
      })

      //information Pays et device emetteur et recepteur
      this.http
    .post<any>(
      "http://localhost:3000/api/findonePays_Devices",{id:idPaysemetteur},
      this.httpOptions
      )
      .subscribe(res=>{
        
           this.infoem=res.data
           this.infoemdev=res.resultat1
           console.log(this.infoemdev)
         
      })
      this.http
      .post<any>(
        "http://localhost:3000/api/findonePays_Devices",{id:idPaysRecepteur},
        this.httpOptions
        )
        .subscribe(res=>{
          
             this.inforec=res.data
             this.inforecdev=res.resultat1
            
             console.log(this.inforecdev)
            
        })
      const swalWithBootstrapButtons = Swal.mixin({
        customClass: {
          confirmButton: 'btn btn-success',
          cancelButton: 'btn btn-danger'
        },
        buttonsStyling: false
      })
      
      swalWithBootstrapButtons.fire({
        title: 'Vous Voulez envoyer à '+prenom+' '+nom+' avec comme CNI '+cni+'Somme envoyé: '+montantenvoye+' montant reçu '+montantreçu+'',
        text: "Vous pouvez annulez d'ici 40 secondes sinon la transaction est confirmé",
        icon: 'warning',
        timer:40000,
        showCancelButton: true,
        confirmButtonText: 'confirmer la transaction',
        cancelButtonText: 'No,Annuler',
        reverseButtons: true
      }).then((result) => {
        if (result.dismiss === Swal.DismissReason.timer) {
          this.router.navigate(['/'])
        }
        if (result.isConfirmed) {
          let frais=montantenvoye*0.01
          console.log('hello suis la'+this.idEmetteur);
          
          this.http
          .post<any>(
            "http://localhost:3000/api/InsertTransaction",{montant_a_recevoir:montantreçu,montantTotal:montantenvoye,status:'envoye',paysDest:this.inforec.nom_pays,paysOrigine:this.infoem.nom_pays,DeviceDest:this.inforecdev.nom_devise,DeviceOrigine:this.infoemdev.nom_devise,frais:frais,DEVISEId:this.infoem.DEVISEId,CLIENTId:+this.idEmetteur,UserId:sessionStorage.getItem('iduser'),recepteurid:+this.idRecepteur},
            this.httpOptions
            )
            .subscribe(res=>{
              if(res.erreur=='false'){
                Swal.fire(
                  'Transaction!',
                  'faite avec succes <a href=/>accueil</a> ou <a href=send>nouvelleTransaction</a>!',
                  'success',
                  
                )
              }
            })
        
        } else if (
          /* Read more about handling dismissals below */
          result.dismiss === Swal.DismissReason.cancel
        ) {
          swalWithBootstrapButtons.fire(
            'Transaction annuler<br> <a href=/>Accueil</a>',
            'Vous Pouvez faire une nouvelle Transaction',
            'error'
          )
        }
      })
      
   }
   calculFrais(event:MouseEvent){
    const index : number=+(event.target as HTMLInputElement).value;
    
   
  
  };
}